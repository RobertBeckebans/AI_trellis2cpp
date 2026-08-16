// dual_grid_cli — mesh a dumped flexible-dual-grid decoder output into a
// T2MESH01 blob.
//
//   usage: dual_grid_cli <in.fdgvox> <out.t2mesh> [--no-cleanup]
//
// The input is what scripts/ref_generate.py writes: the raw 7-channel output of
// the shape-SLAT decoder plus its voxel coordinates. Existing to serve the
// backend-parity comparison — the PyTorch reference cannot mesh its own output
// on Windows (o_voxel is a CUDA extension), so the reference's voxels go
// through OUR extractor, the same one every backend uses. What is left over
// between the resulting mesh and a trellis2.cpp generation is then the network,
// not the mesher.
//
// The post-extraction steps deliberately mirror trellis2_capi.cpp's generate
// path exactly (drop_small_components, grid-scaled fill_holes, vertex_normals,
// orient_faces) — a reference mesh that skipped them would differ from every
// generation for a reason that has nothing to do with the backends.
//
// The topology counters are printed unconditionally: this tool exists to
// produce a comparison, and the comparison is those numbers.

#include "flexible_dual_grid.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

// FDGVOX01: magic[8] | u32 n | u32 grid_size | f32 margin
//           i32[3n] coords | f32[7n] feats   (all little-endian)
bool read_fdgvox( const std::string& path, std::vector<float>& feats, std::vector<int32_t>& coords, int& grid, float& margin )
{
	std::ifstream f( path, std::ios::binary );
	if( !f ) {
		std::fprintf( stderr, "cannot open %s\n", path.c_str() );
		return false;
	}
	char magic[8];
	f.read( magic, 8 );
	if( !f || std::memcmp( magic, "FDGVOX01", 8 ) != 0 ) {
		std::fprintf( stderr, "%s: bad magic (expected FDGVOX01)\n", path.c_str() );
		return false;
	}
	uint32_t n = 0, g = 0;
	float	 m = 0.0f;
	f.read( ( char* )&n, 4 );
	f.read( ( char* )&g, 4 );
	f.read( ( char* )&m, 4 );
	if( !f || n == 0 || g == 0 ) {
		std::fprintf( stderr, "%s: empty or malformed header (n=%u grid=%u)\n", path.c_str(), n, g );
		return false;
	}
	coords.resize( ( size_t )n * 3 );
	feats.resize( ( size_t )n * 7 );
	f.read( ( char* )coords.data(), ( std::streamsize )coords.size() * 4 );
	f.read( ( char* )feats.data(), ( std::streamsize )feats.size() * 4 );
	if( !f ) {
		std::fprintf( stderr, "%s: truncated payload\n", path.c_str() );
		return false;
	}
	grid   = ( int )g;
	margin = m;
	return true;
}

// Same counters trellis2_capi.cpp prints under TRELLIS2_TIMING, so the numbers
// are directly comparable with a generation's log line.
void report_topology( const fdg::Mesh& m, int grid, int fill_limit )
{
	std::unordered_map<uint64_t, int> edge_use;
	edge_use.reserve( m.tris.size() );
	auto ekey = []( int a, int b ) {
		const uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
		return ( ( uint64_t )lo << 32 ) | hi;
	};
	for( size_t t = 0; t + 2 < m.tris.size(); t += 3 ) {
		edge_use[ekey( m.tris[t], m.tris[t + 1] )]++;
		edge_use[ekey( m.tris[t + 1], m.tris[t + 2] )]++;
		edge_use[ekey( m.tris[t + 2], m.tris[t] )]++;
	}
	size_t boundary = 0, nonmanifold = 0;
	for( const auto& e : edge_use ) {
		if( e.second == 1 )
			++boundary;
		else if( e.second > 2 )
			++nonmanifold;
	}
	std::fprintf( stderr,
		"[mesh] grid %d: %zu verts, %zu tris, %zu boundary edges (%.4f%% of %zu), %zu non-manifold edges, fill limit %d\n",
		grid,
		m.n_verts(),
		m.n_tris(),
		boundary,
		edge_use.empty() ? 0.0 : 100.0 * ( double )boundary / ( double )edge_use.size(),
		edge_use.size(),
		nonmanifold,
		fill_limit );
}

// T2MESH01: magic[8] u32 nv u32 nt f32[3nv] verts f32[3nv] normals i32[3nt] tris
bool write_t2mesh( const std::string& path, const fdg::Mesh& m, const std::vector<float>& normals )
{
	std::ofstream f( path, std::ios::binary );
	if( !f ) {
		std::fprintf( stderr, "cannot write %s\n", path.c_str() );
		return false;
	}
	const uint32_t nv = ( uint32_t )m.n_verts();
	const uint32_t nt = ( uint32_t )m.n_tris();
	f.write( "T2MESH01", 8 );
	f.write( ( const char* )&nv, 4 );
	f.write( ( const char* )&nt, 4 );
	f.write( ( const char* )m.verts.data(), ( std::streamsize )m.verts.size() * 4 );
	f.write( ( const char* )normals.data(), ( std::streamsize )normals.size() * 4 );
	f.write( ( const char* )m.tris.data(), ( std::streamsize )m.tris.size() * 4 );
	return ( bool )f;
}

} // namespace

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <in.fdgvox> <out.t2mesh> [--no-cleanup]\n", argv[0] );
		return 2;
	}
	const std::string in_path  = argv[1];
	const std::string out_path = argv[2];
	bool			  cleanup  = true;
	for( int i = 3; i < argc; ++i ) {
		if( std::strcmp( argv[i], "--no-cleanup" ) == 0 )
			cleanup = false;
		else {
			std::fprintf( stderr, "unknown argument: %s\n", argv[i] );
			return 2;
		}
	}

	std::vector<float>	 feats;
	std::vector<int32_t> coords;
	int					 grid	= 0;
	float				 margin = 0.5f;
	if( !read_fdgvox( in_path, feats, coords, grid, margin ) )
		return 1;
	const int nvox = ( int )( coords.size() / 3 );
	std::printf( "%s: %d voxels, grid %d^3, margin %.3f\n", in_path.c_str(), nvox, grid, margin );

	fdg::Mesh mesh = fdg::extract( feats.data(), coords.data(), nvox, grid, margin );
	if( mesh.verts.empty() ) {
		std::fprintf( stderr, "empty mesh (dual grid found no faces)\n" );
		return 1;
	}
	std::printf( "extracted: %zu verts, %zu tris\n", mesh.n_verts(), mesh.n_tris() );

	const int fill_limit = 64 > 64 * grid / 512 ? 64 : 64 * grid / 512;
	if( cleanup ) {
		fdg::drop_small_components( mesh );
		fdg::fill_holes( mesh, fill_limit );
	}
	report_topology( mesh, grid, cleanup ? fill_limit : 0 );

	std::vector<float> normals = fdg::vertex_normals( mesh );
	const size_t	   rewound = fdg::orient_faces( mesh, normals );
	std::printf( "oriented: %zu triangles rewound\n", rewound );

	if( !write_t2mesh( out_path, mesh, normals ) )
		return 1;
	std::printf( "wrote %s (%zu verts, %zu tris)\n", out_path.c_str(), mesh.n_verts(), mesh.n_tris() );
	return 0;
}
