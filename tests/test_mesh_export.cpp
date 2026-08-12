#include "mesh_export.h"
#include "print_remesh.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main()
{
	const float verts[] = {
		0,
		0,
		0,
		1,
		0,
		0,
		0,
		1,
		0,
		0,
		0,
		1,
	};
	const int32_t tris[] = {
		0,
		2,
		1,
		0,
		1,
		3,
		0,
		3,
		2,
		1,
		2,
		3,
	};
	// base RGB, metallic, roughness, alpha
	const float pbr[] = {
		1,
		0,
		0,
		0,
		0.4f,
		0.5f,
		0,
		1,
		0,
		0,
		0.5f,
		1.0f,
		0,
		0,
		1,
		1,
		0.6f,
		1.0f,
		1,
		1,
		1,
		0,
		0.7f,
		1.0f,
	};
	t2glb::MeshExportOptions opt;
	opt.texture_size = 64;
	opt.dilate		 = 2;
	std::vector<uint8_t> glb;
	std::string			 err;
	if( !t2glb::mesh_to_glb( verts, 4, tris, 4, pbr, opt, glb, err ) ) {
		std::fprintf( stderr, "mesh_to_glb failed: %s\n", err.c_str() );
		return 1;
	}
	if( glb.size() < 20 || std::memcmp( glb.data(), "glTF", 4 ) != 0 ) {
		std::fprintf( stderr, "invalid GLB header\n" );
		return 1;
	}
	const std::string bytes( ( const char* )glb.data(), glb.size() );
	if( bytes.find( "\"alphaMode\":\"BLEND\"" ) == std::string::npos ) {
		std::fprintf( stderr, "alpha material was not preserved\n" );
		return 1;
	}
	if( bytes.find( "\"COLOR_0\":2" ) == std::string::npos || bytes.find( "\"_METALLIC_ROUGHNESS\":3" ) == std::string::npos || bytes.find( "\"baseColorTexture\"" ) != std::string::npos ) {
		std::fprintf( stderr, "portable vertex material attributes are missing\n" );
		return 1;
	}
	// The old per-triangle grid duplicated all three vertices and collapsed to
	// sub-texel UV cells on production meshes. The direct path must retain the
	// tetrahedron's four vertices and normalised RGBA bytes exactly.
	auto u32 = [&]( size_t off ) {
		return ( uint32_t )( uint8_t )glb[off] | ( ( uint32_t )( uint8_t )glb[off + 1] << 8 ) | ( ( uint32_t )( uint8_t )glb[off + 2] << 16 ) | ( ( uint32_t )( uint8_t )glb[off + 3] << 24 );
	};
	const size_t  bin					 = 20 + u32( 12 ) + 8;
	const size_t  color					 = bin + 4 * 3 * sizeof( float ) * 2;
	const uint8_t expected_first_color[] = { 255, 255, 0, 0, 0, 0, 0, 128 };
	if( color + sizeof expected_first_color > glb.size() || std::memcmp( glb.data() + color, expected_first_color, sizeof expected_first_color ) != 0 ||
		bytes.find( "\"count\":4,\"type\":\"VEC4\"" ) == std::string::npos ) {
		std::fprintf( stderr, "vertex colours were changed or vertices were duplicated\n" );
		return 1;
	}

	// A tiny near-opaque decoder outlier must not put the whole primitive into
	// alpha blending (which disables depth writes and resembles missing faces).
	float nearly_opaque[24];
	std::memcpy( nearly_opaque, pbr, sizeof nearly_opaque );
	for( int i = 0; i < 4; ++i )
		nearly_opaque[6 * i + 5] = 1.0f;
	nearly_opaque[5] = 0.994f;
	if( !t2glb::mesh_to_glb( verts, 4, tris, 4, nearly_opaque, opt, glb, err ) || std::string( ( const char* )glb.data(), glb.size() ).find( "\"alphaMode\":\"BLEND\"" ) != std::string::npos ) {
		std::fprintf( stderr, "near-opaque PBR noise enabled alpha blending\n" );
		return 1;
	}

	// Preview preparation uses the same geometry path and can keep only the
	// largest disconnected component for background-plane cleanup.
	const float two_component_verts[] = {
		0,
		0,
		0,
		1,
		0,
		0,
		0,
		1,
		0,
		0,
		0,
		1,
		3,
		0,
		0,
		4,
		0,
		0,
		3,
		1,
		0,
	};
	const int32_t two_component_tris[] = {
		0,
		2,
		1,
		0,
		1,
		3,
		0,
		3,
		2,
		1,
		2,
		3,
		4,
		5,
		6,
	};
	t2glb::PreparedMesh prepared;
	opt.components = t2glb::ComponentFilter::KeepLargest;
	if( !t2glb::prepare_mesh( two_component_verts, 7, two_component_tris, 5, nullptr, opt, prepared, err ) ) {
		std::fprintf( stderr, "prepare_mesh failed: %s\n", err.c_str() );
		return 1;
	}
	if( prepared.tris.size() / 3 != 4 || prepared.verts.size() / 3 != 4 || prepared.normals.size() != prepared.verts.size() ) {
		std::fprintf( stderr, "largest-component preview is wrong: %zu verts, %zu tris, %zu normals\n", prepared.verts.size() / 3, prepared.tris.size() / 3, prepared.normals.size() / 3 );
		return 1;
	}

	// Keeping all components must retain every valid source triangle; export no
	// longer performs polygon decimation.
	opt.components = t2glb::ComponentFilter::KeepAll;
	if( !t2glb::prepare_mesh( two_component_verts, 7, two_component_tris, 5, nullptr, opt, prepared, err ) || prepared.tris.size() / 3 != 5 ) {
		std::fprintf( stderr, "full-density export changed the polygon count\n" );
		return 1;
	}

	// The projected bake used to require CGAL and was asserted here to fail
	// without it. Since the tinybvh closest-surface backend landed it works in
	// every build, so the inverse is now the regression to guard: a CGAL-free
	// build must produce a real textured GLB, not an "unavailable" error.
	// Only alpha_wrap still depends on CGAL.
	if( !t2glb::mesh_to_projected_glb( verts, 4, tris, 4, verts, 4, tris, 4, pbr, opt, glb, err ) ) {
		std::fprintf( stderr, "projected bake failed (backend %s): %s\n", t2print::projection_backend(), err.c_str() );
		return 1;
	}
	if( glb.size() < 20 || std::memcmp( glb.data(), "glTF", 4 ) != 0 ) {
		std::fprintf( stderr, "projected bake returned an invalid GLB\n" );
		return 1;
	}
	std::puts( "RESULT: PASS" );
	return 0;
}
