// test_marching_cubes — validate the self-contained isosurface extractor
// (examples/marching_cubes.h) on analytic fields, with no model needed.
//
// Marching tetrahedra on the Freudenthal subdivision must produce a watertight
// 2-manifold: every undirected edge is shared by exactly two triangles, and the
// Euler characteristic V - E + F equals 2*(#components) for closed genus-0
// surfaces. These invariants fail loudly if the tetra triangle table (TET_TRI)
// has a typo, so they double as a table-integrity check.

#include "marching_cubes.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

struct Stats {
	size_t	bad_edges = 0;
	size_t	bad_dir	  = 0;
	int64_t euler	  = 0;
	size_t	E		  = 0;
};

static Stats analyze( const mc::Mesh& m )
{
	auto ukey = []( int a, int b ) -> int64_t {
		int lo = a < b ? a : b, hi = a < b ? b : a;
		return ( int64_t )lo * 100000000LL + hi;
	};
	auto							 dkey = []( int a, int b ) -> int64_t { return ( int64_t )a * 100000000LL + b; };
	std::unordered_map<int64_t, int> uedges, dedges;
	for( size_t t = 0; t < m.n_tris(); ++t ) {
		int v[3] = { m.tris[3 * t], m.tris[3 * t + 1], m.tris[3 * t + 2] };
		for( int i = 0; i < 3; ++i ) {
			int a = v[i], b = v[( i + 1 ) % 3];
			uedges[ukey( a, b )]++;
			dedges[dkey( a, b )]++;
		}
	}
	Stats s;
	s.E = uedges.size();
	for( auto& e : uedges )
		if( e.second != 2 )
			++s.bad_edges;
	// Coherent orientation: each directed edge appears exactly once (its mate is
	// the reverse, contributed by the neighboring triangle). A flipped face makes
	// some directed edge appear twice -> caught here.
	for( auto& e : dedges )
		if( e.second != 1 )
			++s.bad_dir;
	s.euler = ( int64_t )m.n_verts() - ( int64_t )s.E + ( int64_t )m.n_tris();
	return s;
}

static int check( const char* name, const mc::Mesh& m, int64_t want_euler )
{
	Stats s = analyze( m );
	std::printf( "%-16s V=%-6zu E=%-6zu F=%-6zu  bad_edges=%zu bad_winding=%zu  euler=%lld (want %lld)  ",
		name,
		m.n_verts(),
		s.E,
		m.n_tris(),
		s.bad_edges,
		s.bad_dir,
		( long long )s.euler,
		( long long )want_euler );
	bool ok = ( m.n_tris() > 0 ) && ( s.bad_edges == 0 ) && ( s.bad_dir == 0 ) && ( s.euler == want_euler );
	std::printf( "%s\n", ok ? "OK" : "FAIL" );
	return ok ? 0 : 1;
}

int main()
{
	int fails = 0;

	// 1) single sphere -> closed genus-0 manifold, euler 2
	{
		const int		   N = 32;
		const float		   c = ( N - 1 ) * 0.5f, r = 10.0f;
		std::vector<float> f( ( size_t )N * N * N );
		for( int z = 0; z < N; ++z )
			for( int y = 0; y < N; ++y )
				for( int x = 0; x < N; ++x ) {
					float d												   = std::sqrt( ( x - c ) * ( x - c ) + ( y - c ) * ( y - c ) + ( z - c ) * ( z - c ) );
					f[( size_t )x + ( size_t )y * N + ( size_t )z * N * N] = r - d; // inside (>0) within radius
				}
		mc::Mesh m = mc::extract( f.data(), N, N, N, 0.0f );
		fails += check( "sphere", m, 2 );

		// vertex normals must point outward: n . (p - center) > 0
		size_t bad_n = 0;
		for( size_t i = 0; i < m.n_verts(); ++i ) {
			float px = m.verts[3 * i] - c, py = m.verts[3 * i + 1] - c, pz = m.verts[3 * i + 2] - c;
			float d = m.normals[3 * i] * px + m.normals[3 * i + 1] * py + m.normals[3 * i + 2] * pz;
			if( d <= 0.0f )
				++bad_n;
		}
		// FACE winding must point outward too: (P1-P0)x(P2-P0) . (centroid-center) > 0
		size_t bad_f = 0;
		for( size_t t = 0; t < m.n_tris(); ++t ) {
			const float *P0 = &m.verts[3 * m.tris[3 * t]], *P1 = &m.verts[3 * m.tris[3 * t + 1]], *P2 = &m.verts[3 * m.tris[3 * t + 2]];
			float		 u[3] = { P1[0] - P0[0], P1[1] - P0[1], P1[2] - P0[2] }, v[3] = { P2[0] - P0[0], P2[1] - P0[1], P2[2] - P0[2] };
			float		 fn[3] = { u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };
			float		 cx = ( P0[0] + P1[0] + P2[0] ) / 3 - c, cy = ( P0[1] + P1[1] + P2[1] ) / 3 - c, cz = ( P0[2] + P1[2] + P2[2] ) / 3 - c;
			if( fn[0] * cx + fn[1] * cy + fn[2] * cz <= 0.0f )
				++bad_f;
		}
		std::printf( "%-16s inward_vnormals=%zu  inward_faces=%zu  (of %zu)  %s\n", "sphere/orient", bad_n, bad_f, m.n_verts(), ( bad_n == 0 && bad_f == 0 ) ? "OK" : "FAIL" );
		if( bad_n != 0 || bad_f != 0 )
			++fails;
	}

	// 2) sphere that reaches the grid boundary -> still closed thanks to padding
	{
		const int		   N = 24;
		const float		   c = ( N - 1 ) * 0.5f, r = 14.0f; // r > c, clipped by walls
		std::vector<float> f( ( size_t )N * N * N );
		for( int z = 0; z < N; ++z )
			for( int y = 0; y < N; ++y )
				for( int x = 0; x < N; ++x ) {
					float d												   = std::sqrt( ( x - c ) * ( x - c ) + ( y - c ) * ( y - c ) + ( z - c ) * ( z - c ) );
					f[( size_t )x + ( size_t )y * N + ( size_t )z * N * N] = r - d;
				}
		mc::Mesh m = mc::extract( f.data(), N, N, N, 0.0f );
		// clipped sphere is still a closed genus-0 blob -> euler 2
		fails += check( "clipped-sphere", m, 2 );
	}

	// 3) two disjoint spheres -> two components, euler 4
	{
		const int		   N	 = 48;
		const float		   r	 = 7.0f;
		const float		   c1[3] = { 12, 24, 24 }, c2[3] = { 36, 24, 24 };
		std::vector<float> f( ( size_t )N * N * N );
		for( int z = 0; z < N; ++z )
			for( int y = 0; y < N; ++y )
				for( int x = 0; x < N; ++x ) {
					auto sd = [&]( const float c[3] ) { return r - std::sqrt( ( x - c[0] ) * ( x - c[0] ) + ( y - c[1] ) * ( y - c[1] ) + ( z - c[2] ) * ( z - c[2] ) ); };
					f[( size_t )x + ( size_t )y * N + ( size_t )z * N * N] = std::fmax( sd( c1 ), sd( c2 ) );
				}
		mc::Mesh m = mc::extract( f.data(), N, N, N, 0.0f );
		fails += check( "two-spheres", m, 4 );
	}

	std::printf( "\nRESULT: %s\n", fails == 0 ? "PASS" : "FAIL" );
	return fails == 0 ? 0 : 1;
}
