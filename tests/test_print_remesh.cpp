#include "mesh_export.h"
#include "print_remesh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <utility>

struct EdgeUse {
	int count	  = 0;
	int direction = 0;
};

int main()
{
	if( !t2glb::print_remesh_available() ) {
		std::fprintf( stderr, "CGAL print-remesh test was built without its backend\n" );
		return 77;
	}

	// A single open, zero-thickness triangle is deliberately not printable.
	// Alpha Wrap must enclose it in a closed volume despite having no usable
	// source connectivity or inside/outside orientation.
	const float	  verts[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
	const int32_t tris[]  = { 0, 1, 2 };
	const float	  pbr[]	  = {
		1,
		0,
		0,
		0,
		0.5f,
		1,
		0,
		1,
		0,
		0,
		0.5f,
		1,
		0,
		0,
		1,
		0,
		0.5f,
		1,
	};
	t2glb::MeshExportOptions opt;
	opt.components = t2glb::ComponentFilter::KeepAll;
	t2glb::PreparedMesh out;
	std::string			err;
	if( !t2glb::prepare_print_mesh( verts, 3, tris, 1, pbr, opt, 0.20f, 0.03f, out, err ) ) {
		std::fprintf( stderr, "prepare_print_mesh failed: %s\n", err.c_str() );
		return 1;
	}
	if( out.verts.empty() || out.tris.empty() || out.normals.size() != out.verts.size() ) {
		std::fprintf( stderr, "empty/incomplete wrap: %zu verts %zu tris %zu normals\n", out.verts.size() / 3, out.tris.size() / 3, out.normals.size() / 3 );
		return 1;
	}
	// Alpha Wrap builds entirely new vertices, so source per-vertex material
	// cannot be carried over directly; prepare_print_mesh samples it onto them
	// by closest-surface projection for the preview. Every wrap vertex must get
	// a value, and the source's constant metal/rough/alpha must survive the
	// transfer unchanged.
	if( out.pbr.size() != out.verts.size() / 3 * 6 ) {
		std::fprintf( stderr, "preview PBR not projected onto every wrap vertex: %zu vs %zu\n", out.pbr.size() / 6, out.verts.size() / 3 );
		return 1;
	}
	for( size_t v = 0; v < out.pbr.size(); v += 6 ) {
		if( std::fabs( out.pbr[v + 3] - 0.0f ) > 1e-5f || std::fabs( out.pbr[v + 4] - 0.5f ) > 1e-5f || std::fabs( out.pbr[v + 5] - 1.0f ) > 1e-5f ) {
			std::fprintf( stderr, "projected preview material at vertex %zu: %g %g %g\n", v / 6, out.pbr[v + 3], out.pbr[v + 4], out.pbr[v + 5] );
			return 1;
		}
	}

	// Closest-surface transfer must use barycentric interpolation on the source
	// triangle, not nearest-vertex colors. The query is above (0.25,0.25,0),
	// whose expected RGB weights are (0.5,0.25,0.25).
	const std::vector<float>   source_verts( verts, verts + 9 );
	const std::vector<int32_t> source_tris( tris, tris + 3 );
	const std::vector<float>   source_pbr( pbr, pbr + 18 );
	const std::vector<float>   queries = { 0.25f, 0.25f, 0.5f };
	std::vector<float>		   projected;
	if( !t2print::project_pbr( source_verts, source_tris, source_pbr, queries, projected, err ) || projected.size() != 6 ) {
		std::fprintf( stderr, "project_pbr failed: %s\n", err.c_str() );
		return 1;
	}
	const float expected[] = { 0.5f, 0.25f, 0.25f, 0.0f, 0.5f, 1.0f };
	for( int i = 0; i < 6; ++i ) {
		if( std::fabs( projected[i] - expected[i] ) > 1e-5f ) {
			std::fprintf( stderr, "project_pbr channel %d: got %.7g expected %.7g\n", i, projected[i], expected[i] );
			return 1;
		}
	}

	// Exercise the complete target unwrap -> per-texel source projection ->
	// PBR PNG GLB path independently of the legacy T2GLB_XATLAS switch.
	opt.texture_size = 64;
	opt.dilate		 = 2;
	std::vector<uint8_t> glb;
	if( !t2glb::mesh_to_projected_glb( out.verts.data(), ( int )out.verts.size() / 3, out.tris.data(), ( int )out.tris.size() / 3, verts, 3, tris, 1, pbr, opt, glb, err ) ) {
		std::fprintf( stderr, "mesh_to_projected_glb failed: %s\n", err.c_str() );
		return 1;
	}
	if( glb.size() < 20 || std::memcmp( glb.data(), "glTF", 4 ) != 0 ) {
		std::fprintf( stderr, "projected bake returned an invalid GLB\n" );
		return 1;
	}
	const std::string glb_bytes( ( const char* )glb.data(), glb.size() );
	if( glb_bytes.find( "\"baseColorTexture\"" ) == std::string::npos || glb_bytes.find( "\"metallicRoughnessTexture\"" ) == std::string::npos ||
		glb_bytes.find( "\"TEXCOORD_0\"" ) == std::string::npos || glb_bytes.find( "\"COLOR_0\"" ) != std::string::npos ) {
		std::fprintf( stderr, "projected GLB is missing its UV PBR textures\n" );
		return 1;
	}

	std::unordered_map<uint64_t, EdgeUse> edges;
	double								  signed_volume_6 = 0.0;
	for( size_t t = 0; t < out.tris.size(); t += 3 ) {
		const int32_t a = out.tris[t], b = out.tris[t + 1], c = out.tris[t + 2];
		for( const auto e : { std::pair<int32_t, int32_t> { a, b }, { b, c }, { c, a } } ) {
			const uint32_t lo  = ( uint32_t )std::min( e.first, e.second );
			const uint32_t hi  = ( uint32_t )std::max( e.first, e.second );
			EdgeUse&	   use = edges[( ( uint64_t )lo << 32 ) | hi];
			use.count++;
			use.direction += e.first < e.second ? 1 : -1;
		}
		const float* va = out.verts.data() + ( size_t )a * 3;
		const float* vb = out.verts.data() + ( size_t )b * 3;
		const float* vc = out.verts.data() + ( size_t )c * 3;
		signed_volume_6 += va[0] * ( vb[1] * vc[2] - vb[2] * vc[1] ) + va[1] * ( vb[2] * vc[0] - vb[0] * vc[2] ) + va[2] * ( vb[0] * vc[1] - vb[1] * vc[0] );
	}
	for( const auto& it : edges ) {
		if( it.second.count != 2 || it.second.direction != 0 ) {
			std::fprintf( stderr, "non-manifold/inconsistently wound edge: count=%d direction=%d\n", it.second.count, it.second.direction );
			return 1;
		}
	}
	if( std::fabs( signed_volume_6 ) < 1e-8 ) {
		std::fprintf( stderr, "Alpha Wrap did not enclose a non-zero volume\n" );
		return 1;
	}

	std::printf( "RESULT: PASS (%zu verts, %zu tris)\n", out.verts.size() / 3, out.tris.size() / 3 );
	return 0;
}
