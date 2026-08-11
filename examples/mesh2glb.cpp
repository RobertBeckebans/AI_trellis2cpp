// mesh2glb — export a demo mesh (T2MESH01/T2MESH02/T2MESH03 wire format) to a
// portable vertex-coloured GLB. Exercises the CUDA-free component cleanup and
// glTF path offline, with no models or GPU. T2GLB_XATLAS opts into image baking.
//
//   mesh2glb in.bin out.glb [texture_size] [--print [alpha_pct offset_pct]]
//
// The wire format is what the demo server emits at /api/mesh/{id}:
//   magic[8]  u32 nv  u32 nt  f32[3nv] verts  f32[3nv] normals
//   [T2MESH02: f32[5nv] legacy pbr]
//   [T2MESH03: f32[6nv] pbr incl. alpha] i32[3nt] tris (little-endian)

#include "mesh_export.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

template<class T>
static bool rd( FILE* f, std::vector<T>& v, size_t n )
{
	v.resize( n );
	return n == 0 || std::fread( v.data(), sizeof( T ), n, f ) == n;
}

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s in.bin out.glb [texture_size] [--print [alpha_pct offset_pct]]\n", argv[0] );
		return 2;
	}
	t2glb::MeshExportOptions opt;
	opt.components	  = t2glb::ComponentFilter::KeepAll;
	bool  print_wrap  = false;
	float alpha_ratio = 0.01f, offset_ratio = 0.01f / 30.0f;
	for( int i = 3; i < argc; ++i ) {
		if( std::strcmp( argv[i], "--print" ) == 0 ) {
			print_wrap = true;
			if( i + 1 < argc && argv[i + 1][0] != '-' )
				alpha_ratio = std::atof( argv[++i] ) / 100.0f;
			if( i + 1 < argc && argv[i + 1][0] != '-' )
				offset_ratio = std::atof( argv[++i] ) / 100.0f;
		} else {
			opt.texture_size = std::atoi( argv[i] );
		}
	}
	if( print_wrap && !t2glb::print_remesh_available() ) {
		std::fprintf( stderr, "print remeshing unavailable: rebuild with CGAL 5.5 or newer\n" );
		return 1;
	}

	FILE* f = std::fopen( argv[1], "rb" );
	if( !f ) {
		std::fprintf( stderr, "open %s failed\n", argv[1] );
		return 1;
	}
	char	 magic[9] = { 0 };
	uint32_t nv = 0, nt = 0;
	if( std::fread( magic, 1, 8, f ) != 8 || std::fread( &nv, 4, 1, f ) != 1 || std::fread( &nt, 4, 1, f ) != 1 ) {
		std::fprintf( stderr, "bad header\n" );
		return 1;
	}
	const bool legacy	= std::memcmp( magic, "T2MESH02", 8 ) == 0;
	const bool textured = legacy || std::memcmp( magic, "T2MESH03", 8 ) == 0;
	if( !textured && std::memcmp( magic, "T2MESH01", 8 ) != 0 ) {
		std::fprintf( stderr, "unknown magic\n" );
		return 1;
	}
	std::vector<float>	 verts, normals, pbr;
	std::vector<int32_t> tris;
	bool				 ok = rd( f, verts, ( size_t )nv * 3 ) && rd( f, normals, ( size_t )nv * 3 );
	if( ok && textured ) {
		if( legacy ) {
			std::vector<float> old;
			ok = rd( f, old, ( size_t )nv * 5 );
			if( ok ) {
				pbr.resize( ( size_t )nv * 6 );
				for( uint32_t i = 0; i < nv; ++i ) {
					std::memcpy( pbr.data() + ( size_t )i * 6, old.data() + ( size_t )i * 5, 5 * sizeof( float ) );
					pbr[( size_t )i * 6 + 5] = 1.0f;
				}
			}
		} else {
			ok = rd( f, pbr, ( size_t )nv * 6 );
		}
	}
	ok = ok && rd( f, tris, ( size_t )nt * 3 );
	std::fclose( f );
	if( !ok ) {
		std::fprintf( stderr, "truncated mesh\n" );
		return 1;
	}

	std::fprintf( stderr, "in: %s  %u verts  %u tris  %s\n", magic, nv, nt, textured ? "textured" : "geometry-only" );
	std::fprintf( stderr, "export: %s ...\n", print_wrap ? "constructing watertight CGAL Alpha Wrap" : "preserving input topology" );

	std::vector<uint8_t> glb;
	std::string			 err;
	t2glb::PreparedMesh	 wrapped;
	const float*		 export_verts = verts.data();
	const int32_t*		 export_tris  = tris.data();
	const float*		 export_pbr	  = textured ? pbr.data() : nullptr;
	int					 export_nv = ( int )nv, export_nt = ( int )nt;
	if( print_wrap ) {
		if( !t2glb::prepare_print_mesh( verts.data(), ( int )nv, tris.data(), ( int )nt, export_pbr, opt, alpha_ratio, offset_ratio, wrapped, err ) ) {
			std::fprintf( stderr, "prepare_print_mesh: %s\n", err.c_str() );
			return 1;
		}
		export_verts   = wrapped.verts.data();
		export_nv	   = ( int )wrapped.verts.size() / 3;
		export_tris	   = wrapped.tris.data();
		export_nt	   = ( int )wrapped.tris.size() / 3;
		export_pbr	   = nullptr;
		opt.components = t2glb::ComponentFilter::KeepAll;
		std::fprintf( stderr, "wrap: %d verts  %d tris  %s\n", export_nv, export_nt, textured ? "rebaking source PBR atlas" : "geometry-only" );
	}
	const bool projected = print_wrap && textured;
	const bool baked	 = projected ? t2glb::mesh_to_projected_glb( export_verts, export_nv, export_tris, export_nt, verts.data(), ( int )nv, tris.data(), ( int )nt, pbr.data(), opt, glb, err ) :
									   t2glb::mesh_to_glb( export_verts, export_nv, export_tris, export_nt, export_pbr, opt, glb, err );
	if( !baked ) {
		std::fprintf( stderr, "%s: %s\n", projected ? "mesh_to_projected_glb" : "mesh_to_glb", err.c_str() );
		return 1;
	}

	FILE* o = std::fopen( argv[2], "wb" );
	if( !o ) {
		std::fprintf( stderr, "open %s failed\n", argv[2] );
		return 1;
	}
	std::fwrite( glb.data(), 1, glb.size(), o );
	std::fclose( o );
	std::fprintf( stderr, "wrote %s (%.2f MB)\n", argv[2], glb.size() / 1048576.0 );
	return 0;
}
