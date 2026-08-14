// Topological invariants of the flexible dual grid (examples/flexible_dual_grid.h)
// on analytic fields. No model, no GPU, no fixtures.
//
// docs/VERIFICATION.md credited "dual-grid mesh extraction" to
// test_marching_cubes, but that test exercises examples/marching_cubes.h — a
// different extractor. fdg::extract, which is what every generated mesh
// actually comes out of, had no test at all.
//
// What is checked:
//   1. extract() on a closed analytic surface is a closed 2-manifold: no
//      boundary edges, no edge shared by more than two triangles, and the Euler
//      characteristic of the genus it was given (2 for a sphere, 0 for a torus).
//   2. The same holds at two grid resolutions, so the extractor is not
//      accidentally scale-dependent.
//   3. drop_small_components() and fill_holes() leave an already-closed mesh
//      closed — they are cleanup, and cleanup that damages a good mesh is worse
//      than none.
//   4. fill_holes()'s contract on a punctured surface: a boundary loop shorter
//      than max_loop is closed, a longer one is left alone. That parameter
//      counts EDGES, so it is not scale-free — which is what made the same
//      physical hole survive at a finer grid until the caller started scaling
//      it. See docs/progress/plan_dual-grid-hole-fill.md.

#include "flexible_dual_grid.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{

int	 n_fail = 0;

void check( bool ok, const char* what )
{
	std::printf( "  [%s] %s\n", ok ? "OK  " : "FAIL", what );
	if( !ok )
		++n_fail;
}

void check_eq( long long got, long long want, const char* what )
{
	const bool ok = got == want;
	std::printf( "  [%s] %s: got %lld, want %lld\n", ok ? "OK  " : "FAIL", what, got, want );
	if( !ok )
		++n_fail;
}

uint64_t ekey( int a, int b )
{
	const uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
	return ( ( uint64_t )lo << 32 ) | hi;
}

struct topology {
	size_t	  tris = 0, edges = 0, boundary = 0, nonmanifold = 0, used_verts = 0;
	long long euler = 0;
};

topology analyse( const fdg::Mesh& m )
{
	topology						  t;
	std::unordered_map<uint64_t, int> use;
	std::unordered_set<int>			  verts;
	use.reserve( m.tris.size() );
	for( size_t i = 0; i + 2 < m.tris.size(); i += 3 ) {
		const int a = m.tris[i], b = m.tris[i + 1], c = m.tris[i + 2];
		use[ekey( a, b )]++;
		use[ekey( b, c )]++;
		use[ekey( c, a )]++;
		verts.insert( a );
		verts.insert( b );
		verts.insert( c );
	}
	t.tris		 = m.tris.size() / 3;
	t.edges		 = use.size();
	t.used_verts = verts.size();
	for( const auto& e : use ) {
		if( e.second == 1 )
			++t.boundary;
		else if( e.second > 2 )
			++t.nonmanifold;
	}
	// Only vertices that carry a triangle count: extract() emits one vertex per
	// input voxel, and the voxels that exist purely so a neighbour's quad can
	// close are isolated in the mesh.
	t.euler = ( long long )t.used_verts - ( long long )t.edges + ( long long )t.tris;
	return t;
}

// Build the dual-grid input for an implicit surface, the way the decoder's
// output is laid out: 7 channels per voxel, [0..2] dual-vertex offset logits,
// [3..5] "this axis' edge is crossed", [6] the split weight.
//
// Voxel (cx,cy,cz) owns the grid edge from corner (cx,cy,cz)+(1,1,1)-e_a along
// +e_a — that is the edge the four cells in EDGE_OFF[a] surround. A quad is
// emitted only when all four of those cells exist, so the voxel set is the
// union of the four-cell neighbourhoods of every crossed edge, not just the
// cells that own one.
struct fixture {
	std::vector<int32_t> coords;
	std::vector<float>	 feats;
};

fixture build( int grid, const std::function<double( double, double, double )>& sdf, const std::function<bool( int, int, int )>& keep )
{
	auto											 inside = [&]( int x, int y, int z ) { return sdf( x, y, z ) < 0.0; };
	// owner voxel -> crossed axes
	std::unordered_map<uint64_t, int>				 axes;
	std::unordered_map<uint64_t, std::array<int, 3>> pos;
	auto											 key = []( int x, int y, int z ) { return ( ( uint64_t )( uint32_t )x << 40 ) | ( ( uint64_t )( uint32_t )y << 20 ) | ( uint64_t )( uint32_t )z; };
	auto											 add = [&]( int x, int y, int z, int axis_bit ) {
		if( x < 0 || y < 0 || z < 0 || x >= grid || y >= grid || z >= grid )
			return;
		const uint64_t k = key( x, y, z );
		axes[k] |= axis_bit;
		pos[k] = { x, y, z };
	};

	for( int x = 0; x < grid; ++x )
		for( int y = 0; y < grid; ++y )
			for( int z = 0; z < grid; ++z ) {
				if( !keep( x, y, z ) )
					continue;
				for( int a = 0; a < 3; ++a ) {
					// edge from c to c + e_a
					int c[3] = { x + ( a != 0 ), y + ( a != 1 ), z + ( a != 2 ) };
					int d[3] = { c[0], c[1], c[2] };
					d[a] += 1;
					if( inside( c[0], c[1], c[2] ) == inside( d[0], d[1], d[2] ) )
						continue;
					add( x, y, z, 1 << a ); // the owner, with this axis crossed
					// ...and the other three cells around the edge, so the quad
					// can close even where they cross nothing themselves.
					for( int i = 1; i < 4; ++i )
						add( x + fdg::detail::EDGE_OFF[a][i][0], y + fdg::detail::EDGE_OFF[a][i][1], z + fdg::detail::EDGE_OFF[a][i][2], 0 );
				}
			}

	fixture f;
	f.coords.reserve( axes.size() * 3 );
	f.feats.reserve( axes.size() * 7 );
	for( const auto& e : axes ) {
		const auto& p = pos[e.first];
		f.coords.push_back( p[0] );
		f.coords.push_back( p[1] );
		f.coords.push_back( p[2] );
		// offsets 0 -> sigmoid 0.5 -> the cell centre. Positions do not affect
		// topology, which is what this test is about.
		for( int a = 0; a < 3; ++a )
			f.feats.push_back( 0.0f );
		for( int a = 0; a < 3; ++a )
			f.feats.push_back( ( e.second & ( 1 << a ) ) ? 1.0f : -1.0f );
		f.feats.push_back( 0.0f ); // split weight, equal everywhere
	}
	return f;
}

void report( const char* name, const topology& t, long long want_euler )
{
	std::printf( "%s: %zu verts used, %zu tris, %zu edges, chi = %lld\n", name, t.used_verts, t.tris, t.edges, t.euler );
	check_eq( ( long long )t.boundary, 0, "no boundary edges (the surface is closed)" );
	check_eq( ( long long )t.nonmanifold, 0, "no edge shared by more than two triangles" );
	check_eq( t.euler, want_euler, "Euler characteristic" );
}

} // namespace

int main()
{
	// ── 1. a sphere, at two grid resolutions ─────────────────────────────────
	for( int grid : { 48, 96 } ) {
		const double c = grid * 0.5, r = grid * 0.3;
		auto		 sdf = [&]( double x, double y, double z ) { return std::sqrt( ( x - c ) * ( x - c ) + ( y - c ) * ( y - c ) + ( z - c ) * ( z - c ) ) - r; };
		fixture		 f	 = build( grid, sdf, []( int, int, int ) { return true; } );
		fdg::Mesh	 m	 = fdg::extract( f.feats.data(), f.coords.data(), ( int )( f.coords.size() / 3 ), grid );
		char		 name[64];
		std::snprintf( name, sizeof( name ), "\nsphere at grid %d", grid );
		report( name, analyse( m ), 2 );

		// Cleanup must not damage a mesh that is already closed.
		fdg::drop_small_components( m );
		fdg::fill_holes( m, 64 * grid / 48 );
		const topology after = analyse( m );
		check_eq( ( long long )after.boundary, 0, "still closed after drop_small_components + fill_holes" );
		check_eq( after.euler, 2, "Euler characteristic unchanged by cleanup" );
	}

	// ── 2. a torus: closed, but genus 1 ──────────────────────────────────────
	{
		const int	 grid = 64;
		const double c = grid * 0.5, R = grid * 0.28, r = grid * 0.12;
		auto		 sdf = [&]( double x, double y, double z ) {
			const double q = std::sqrt( ( x - c ) * ( x - c ) + ( y - c ) * ( y - c ) ) - R;
			return std::sqrt( q * q + ( z - c ) * ( z - c ) ) - r;
		};
		fixture	  f = build( grid, sdf, []( int, int, int ) { return true; } );
		fdg::Mesh m = fdg::extract( f.feats.data(), f.coords.data(), ( int )( f.coords.size() / 3 ), grid );
		report( "\ntorus at grid 64", analyse( m ), 0 );
	}

	// ── 3. fill_holes' contract on a punctured sphere ────────────────────────
	// The limit counts boundary EDGES, so the same physical opening needs a
	// larger limit at a finer grid. That is the whole of the bug this guards.
	{
		const int	   grid = 64;
		const double   c = grid * 0.5, r = grid * 0.3;
		auto		   sdf = [&]( double x, double y, double z ) { return std::sqrt( ( x - c ) * ( x - c ) + ( y - c ) * ( y - c ) + ( z - c ) * ( z - c ) ) - r; };
		// Drop a small cap of owner voxels: their quads vanish and leave one
		// boundary loop.
		auto		   keep	  = [&]( int x, int y, int z ) { return !( z > c + r - 3 && std::abs( x - c ) < 3 && std::abs( y - c ) < 3 ); };
		fixture		   f	  = build( grid, sdf, keep );
		fdg::Mesh	   base	  = fdg::extract( f.feats.data(), f.coords.data(), ( int )( f.coords.size() / 3 ), grid );
		const topology open_t = analyse( base );
		std::printf( "\npunctured sphere at grid 64: %zu boundary edges\n", open_t.boundary );
		check( open_t.boundary > 0, "the puncture really leaves a boundary" );

		fdg::Mesh generous = base;
		fdg::fill_holes( generous, 4096 );
		check_eq( ( long long )analyse( generous ).boundary, 0, "a generous limit closes the puncture" );

		fdg::Mesh stingy = base;
		fdg::fill_holes( stingy, 3 ); // shorter than any real loop
		check( analyse( stingy ).boundary > 0, "a limit below the loop length leaves it open" );
	}

	std::printf( "\ntotal failures: %d\nRESULT: %s\n", n_fail, n_fail ? "FAIL" : "PASS" );
	return n_fail ? 1 : 0;
}
