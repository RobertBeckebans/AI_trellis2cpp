// test_quad_remesh.cpp — invariants of the AutoRemesher quad stage on synthetic
// fixtures. No model assets, no GPU, no network.
//
// The point of this test is *not* to pin the output down: AutoRemesher's frame
// field is free to place its singularities wherever it likes, and it may drop
// islands it fails to parameterize. What must hold is that the face stream we
// hand downstream is structurally valid, that the reported statistics describe
// the mesh actually returned, and that two runs on the same input agree.
//
// Exit:  0 as expected, 1 an invariant broke, 77 built without the backend.

#include "quad_remesh.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

// A closed axis-aligned cube, two triangles per face, wound outward. Twelve
// triangles is what Phase 1 measured, and the coarsest closed input the stage
// is expected to survive.
void make_cube( std::vector<float>& verts, std::vector<int32_t>& tris )
{
	verts = {
		-1,
		-1,
		-1, //
		1,
		-1,
		-1, //
		1,
		1,
		-1, //
		-1,
		1,
		-1, //
		-1,
		-1,
		1, //
		1,
		-1,
		1, //
		1,
		1,
		1, //
		-1,
		1,
		1, //
	};
	tris = {
		0,
		2,
		1,
		0,
		3,
		2, // -Z
		4,
		5,
		6,
		4,
		6,
		7, // +Z
		0,
		1,
		5,
		0,
		5,
		4, // -Y
		3,
		7,
		6,
		3,
		6,
		2, // +Y
		0,
		4,
		7,
		0,
		7,
		3, // -X
		1,
		2,
		6,
		1,
		6,
		5, // +X
	};
}

// A closed UV sphere: curvature for the frame field to align to, and poles as
// the natural singularities. Wound outward, poles shared so it stays manifold.
void make_sphere( int sectors, int rings, std::vector<float>& verts, std::vector<int32_t>& tris )
{
	verts.clear();
	tris.clear();
	const double pi = 3.14159265358979323846;
	verts.insert( verts.end(), { 0.0f, 1.0f, 0.0f } ); // north pole = vertex 0
	for( int r = 1; r < rings; ++r ) {
		const double phi = pi * ( double )r / ( double )rings;
		for( int s = 0; s < sectors; ++s ) {
			const double th = 2.0 * pi * ( double )s / ( double )sectors;
			verts.push_back( ( float )( std::sin( phi ) * std::cos( th ) ) );
			verts.push_back( ( float )std::cos( phi ) );
			verts.push_back( ( float )( std::sin( phi ) * std::sin( th ) ) );
		}
	}
	const int32_t south = ( int32_t )( verts.size() / 3 );
	verts.insert( verts.end(), { 0.0f, -1.0f, 0.0f } );

	auto ring = [&]( int r, int s ) { return ( int32_t )( 1 + ( r - 1 ) * sectors + ( s % sectors ) ); };
	for( int s = 0; s < sectors; ++s ) {
		tris.insert( tris.end(), { 0, ring( 1, s + 1 ), ring( 1, s ) } );
		tris.insert( tris.end(), { south, ring( rings - 1, s ), ring( rings - 1, s + 1 ) } );
	}
	for( int r = 1; r < rings - 1; ++r )
		for( int s = 0; s < sectors; ++s ) {
			const int32_t a = ring( r, s ), b = ring( r, s + 1 );
			const int32_t c = ring( r + 1, s ), d = ring( r + 1, s + 1 );
			tris.insert( tris.end(), { a, b, d } );
			tris.insert( tris.end(), { a, d, c } );
		}
}

// Every structural invariant the face stream must satisfy, plus a check that
// the reported statistics describe *this* mesh and not an earlier one.
bool check_output( const char* what, const std::vector<float>& out_verts, const std::vector<int32_t>& out_faces, const std::vector<int32_t>& out_face_sizes, const t2quad::QuadRemeshStats& stats )
{
	const size_t nv = out_verts.size() / 3;
	if( nv == 0 || out_face_sizes.empty() ) {
		std::fprintf( stderr, "%s: empty output (%zu verts, %zu faces)\n", what, nv, out_face_sizes.size() );
		return false;
	}
	if( out_verts.size() % 3 != 0 ) {
		std::fprintf( stderr, "%s: vertex array is not a multiple of 3\n", what );
		return false;
	}
	for( size_t i = 0; i < out_verts.size(); ++i )
		if( !std::isfinite( out_verts[i] ) ) {
			std::fprintf( stderr, "%s: non-finite coordinate at %zu\n", what, i );
			return false;
		}

	// face_sizes must exactly partition the flat index stream, and no face may
	// have fewer than three corners.
	size_t cursor = 0;
	int	   quads = 0, triangles = 0, ngons = 0;
	for( size_t f = 0; f < out_face_sizes.size(); ++f ) {
		const int n = out_face_sizes[f];
		if( n < 3 ) {
			std::fprintf( stderr, "%s: face %zu has %d corners\n", what, f, n );
			return false;
		}
		if( cursor + ( size_t )n > out_faces.size() ) {
			std::fprintf( stderr, "%s: face %zu runs past the index stream\n", what, f );
			return false;
		}
		for( int k = 0; k < n; ++k ) {
			const int32_t v = out_faces[cursor + k];
			if( v < 0 || ( size_t )v >= nv ) {
				std::fprintf( stderr, "%s: face %zu corner %d indexes vertex %d of %zu\n", what, f, k, v, nv );
				return false;
			}
			// A face that visits the same vertex twice is degenerate, not a
			// legitimate n-gon.
			for( int j = 0; j < k; ++j )
				if( out_faces[cursor + j] == v ) {
					std::fprintf( stderr, "%s: face %zu repeats vertex %d\n", what, f, v );
					return false;
				}
		}
		cursor += ( size_t )n;
		if( n == 3 )
			++triangles;
		else if( n == 4 )
			++quads;
		else
			++ngons;
	}
	if( cursor != out_faces.size() ) {
		std::fprintf( stderr, "%s: %zu indices left over after the last face\n", what, out_faces.size() - cursor );
		return false;
	}
	if( stats.quads != quads || stats.triangles != triangles || stats.ngons != ngons ) {
		std::fprintf( stderr, "%s: stats say %d/%d/%d quads/tris/ngons, the stream has %d/%d/%d\n", what, stats.quads, stats.triangles, stats.ngons, quads, triangles, ngons );
		return false;
	}

	// Quad-dominant is the entire point of the stage; a result that is mostly
	// triangles means the parameterization collapsed and the caller would be
	// better off with the input.
	const double quad_ratio = ( double )quads / ( double )out_face_sizes.size();
	if( quad_ratio < 0.5 ) {
		std::fprintf( stderr, "%s: only %.1f%% quads\n", what, 100.0 * quad_ratio );
		return false;
	}

	// Recount the boundary from the face stream. A non-zero count is a
	// legitimate outcome — AutoRemesher's quad extraction can leave the surface
	// open, which is exactly why quad_remesh.h refuses to promise watertightness
	// and why the server surfaces the number. What is *not* acceptable is the
	// statistic disagreeing with the mesh it describes, because that is what the
	// viewer and the export hints are read off.
	std::unordered_map<uint64_t, int> edges;
	cursor = 0;
	for( const int32_t n : out_face_sizes ) {
		for( int k = 0; k < n; ++k ) {
			const uint32_t a  = ( uint32_t )out_faces[cursor + k];
			const uint32_t b  = ( uint32_t )out_faces[cursor + ( k + 1 ) % n];
			const uint64_t lo = a < b ? a : b, hi = a < b ? b : a;
			edges[( lo << 32 ) | hi]++;
		}
		cursor += ( size_t )n;
	}
	int boundary = 0;
	for( const auto& e : edges )
		if( e.second == 1 )
			++boundary;
	if( stats.boundary_edges != boundary ) {
		std::fprintf( stderr, "%s: stats report %d boundary edges, the stream has %d\n", what, stats.boundary_edges, boundary );
		return false;
	}

	if( !( stats.area_retained > 0.0f ) || !std::isfinite( stats.area_retained ) ) {
		std::fprintf( stderr, "%s: area_retained is %g\n", what, stats.area_retained );
		return false;
	}

	std::printf( "%-14s %5zu verts %5zu faces  %5.1f%% quads  %4d boundary edge(s)  %5.1f%% area\n", what, nv, out_face_sizes.size(), 100.0 * quad_ratio, boundary, 100.0f * stats.area_retained );
	return true;
}

// triangulate() must cover every face exactly once and invent no indices.
bool check_triangulation( const char* what, const std::vector<float>& verts, const std::vector<int32_t>& faces, const std::vector<int32_t>& sizes )
{
	std::vector<int32_t> tris;
	t2quad::triangulate( verts, faces, sizes, tris );
	size_t expected = 0;
	for( const int32_t n : sizes )
		expected += ( size_t )( n - 2 );
	if( tris.size() != expected * 3 ) {
		std::fprintf( stderr, "%s: triangulate produced %zu triangles, expected %zu\n", what, tris.size() / 3, expected );
		return false;
	}
	const size_t nv = verts.size() / 3;
	for( size_t t = 0; t < tris.size(); t += 3 ) {
		for( int k = 0; k < 3; ++k )
			if( tris[t + k] < 0 || ( size_t )tris[t + k] >= nv ) {
				std::fprintf( stderr, "%s: triangulated index %d out of range\n", what, tris[t + k] );
				return false;
			}
		if( tris[t] == tris[t + 1] || tris[t + 1] == tris[t + 2] || tris[t] == tris[t + 2] ) {
			std::fprintf( stderr, "%s: triangulate emitted a degenerate triangle\n", what );
			return false;
		}
	}
	return true;
}

} // namespace

int main()
{
	if( !t2quad::available() ) {
		std::printf( "quad remeshing unavailable (built without TRELLIS2_AUTOREMESHER or Eigen 5.x)\n" );
		return 77;
	}

	std::string				  err;
	t2quad::QuadRemeshOptions opt;
	opt.target_quads = 2000;

	// ---------------------------------------------------------------------
	// Cube: the coarsest closed input, twelve triangles.
	// ---------------------------------------------------------------------
	{
		std::vector<float>	 verts;
		std::vector<int32_t> tris;
		make_cube( verts, tris );

		std::vector<float>		out_verts;
		std::vector<int32_t>	out_faces, out_sizes;
		t2quad::QuadRemeshStats stats;
		if( !t2quad::remesh( verts, tris, opt, out_verts, out_faces, out_sizes, stats, err ) ) {
			std::fprintf( stderr, "cube: remesh failed: %s\n", err.c_str() );
			return 1;
		}
		if( !check_output( "cube", out_verts, out_faces, out_sizes, stats ) )
			return 1;
		if( !check_triangulation( "cube", out_verts, out_faces, out_sizes ) )
			return 1;
	}

	// ---------------------------------------------------------------------
	// Sphere: curvature, and poles where the frame field must place
	// singularities. Also the determinism fixture — two runs on one input must
	// agree bit for bit. The per-island merge is ordered, so this should hold;
	// it is asserted because a silent dependence on thread scheduling would
	// otherwise only surface as a cache key that never hits.
	// ---------------------------------------------------------------------
	{
		std::vector<float>	 verts;
		std::vector<int32_t> tris;
		make_sphere( 32, 16, verts, tris );

		std::vector<float>		out_verts[2];
		std::vector<int32_t>	out_faces[2], out_sizes[2];
		t2quad::QuadRemeshStats stats[2];
		for( int run = 0; run < 2; ++run )
			if( !t2quad::remesh( verts, tris, opt, out_verts[run], out_faces[run], out_sizes[run], stats[run], err ) ) {
				std::fprintf( stderr, "sphere run %d: remesh failed: %s\n", run, err.c_str() );
				return 1;
			}
		if( !check_output( "sphere", out_verts[0], out_faces[0], out_sizes[0], stats[0] ) )
			return 1;
		if( !check_triangulation( "sphere", out_verts[0], out_faces[0], out_sizes[0] ) )
			return 1;

		if( out_verts[0] != out_verts[1] || out_faces[0] != out_faces[1] || out_sizes[0] != out_sizes[1] ) {
			std::fprintf( stderr, "two runs on the same input differ: %zu/%zu verts, %zu/%zu faces\n", out_verts[0].size() / 3, out_verts[1].size() / 3, out_sizes[0].size(), out_sizes[1].size() );
			return 1;
		}
		if( std::memcmp( &stats[0], &stats[1], sizeof( t2quad::QuadRemeshStats ) ) != 0 ) {
			std::fprintf( stderr, "two runs on the same input reported different statistics\n" );
			return 1;
		}
		std::printf( "determinism    two runs on the sphere are byte-identical\n" );
	}

	// ---------------------------------------------------------------------
	// Degenerate inputs must be refused, not remeshed into nonsense.
	// ---------------------------------------------------------------------
	{
		const std::vector<float>   empty_verts;
		const std::vector<int32_t> empty_tris;
		std::vector<float>		   ov;
		std::vector<int32_t>	   of, os;
		t2quad::QuadRemeshStats	   st;
		if( t2quad::remesh( empty_verts, empty_tris, opt, ov, of, os, st, err ) ) {
			std::fprintf( stderr, "an empty mesh was accepted\n" );
			return 1;
		}
	}

	std::printf( "RESULT: PASS\n" );
	return 0;
}
