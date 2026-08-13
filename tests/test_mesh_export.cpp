#include "mesh_export.h"
#include "print_remesh.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

struct GlbView {
	uint32_t off = 0, len = 0;
};

// Minimal reader for the GLBs this module writes, not a general glTF parser.
// The bufferView order is the one write_glb appends its streams in — position,
// normal, uv, tangent, indices, baseColor, metallicRoughness, normal map — and
// the caller checks the JSON markers that prove which of the optional ones are
// present.
bool glb_open( const std::vector<uint8_t>& glb, std::string& json, const uint8_t*& bin, std::vector<GlbView>& views )
{
	views.clear();
	if( glb.size() < 20 || std::memcmp( glb.data(), "glTF", 4 ) != 0 )
		return false;
	auto		   u32		= [&]( size_t off ) { return ( uint32_t )glb[off] | ( ( uint32_t )glb[off + 1] << 8 ) | ( ( uint32_t )glb[off + 2] << 16 ) | ( ( uint32_t )glb[off + 3] << 24 ); };
	const uint32_t json_len = u32( 12 );
	if( 20 + ( size_t )json_len + 8 > glb.size() )
		return false;
	json.assign( ( const char* )glb.data() + 20, json_len );
	bin = glb.data() + 20 + json_len + 8;

	size_t p = json.find( "\"bufferViews\":[" );
	if( p == std::string::npos )
		return false;
	const size_t end = json.find( ']', p );
	while( true ) {
		const size_t o = json.find( "\"byteOffset\":", p );
		if( o == std::string::npos || o > end )
			break;
		const size_t l = json.find( "\"byteLength\":", o );
		if( l == std::string::npos || l > end )
			break;
		GlbView v;
		v.off = ( uint32_t )std::strtoul( json.c_str() + o + 13, nullptr, 10 );
		v.len = ( uint32_t )std::strtoul( json.c_str() + l + 13, nullptr, 10 );
		views.push_back( v );
		p = l;
	}
	return !views.empty();
}

} // namespace

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
	// ---------------------------------------------------------------------
	// Tangent-space normal map round trip.
	//
	// A flat target quad against a *tilted* flat source plane: the source's
	// normal is one known constant, and the target's tangent frame is constant
	// too (one planar chart means one affine UV map). So every baked texel must
	// reconstruct to the same world normal, and that normal must be the
	// source's. This is the check that catches a flipped green channel, a wrong
	// handedness, or a swizzle mistake — each of which looks entirely plausible
	// by eye and only shows up under lighting.
	// ---------------------------------------------------------------------
	{
		t2glb::MeshExportOptions nopt;
		nopt.components	  = t2glb::ComponentFilter::KeepAll;
		nopt.texture_size = 128;
		nopt.dilate		  = 2;

		// Target: unit quad in the z = 0 plane.
		const float	  tv[] = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0 };
		const int32_t tt[] = { 0, 1, 2, 0, 2, 3 };

		// Source: a larger plane tilted about x, so its normal is not the
		// target's and the two axes of the tangent basis are actually exercised.
		const float	  tilt	 = 0.3f;
		const float	  sv[]	 = { -1, -1, 0.15f - tilt, 2, -1, 0.15f - tilt, 2, 2, 0.15f + 2 * tilt, -1, 2, 0.15f + 2 * tilt };
		const int32_t st[]	 = { 0, 1, 2, 0, 2, 3 };
		const float	  spbr[] = {
			  0.8f,
			  0.2f,
			  0.2f,
			  0.0f,
			  0.5f,
			  1.0f,
			  0.8f,
			  0.2f,
			  0.2f,
			  0.0f,
			  0.5f,
			  1.0f,
			  0.8f,
			  0.2f,
			  0.2f,
			  0.0f,
			  0.5f,
			  1.0f,
			  0.8f,
			  0.2f,
			  0.2f,
			  0.0f,
			  0.5f,
			  1.0f,
		};

		// The normal the bake will actually see, taken from the same preparation
		// the bake runs rather than assumed from the winding.
		t2glb::PreparedMesh src;
		if( !t2glb::prepare_mesh( sv, 4, st, 2, spbr, nopt, src, err ) || src.normals.size() < 3 ) {
			std::fprintf( stderr, "could not prepare the tilted source: %s\n", err.c_str() );
			return 1;
		}
		// TRELLIS -> glTF Y-up, the same rotation write_glb applies.
		const float			 want[3] = { src.normals[0], src.normals[2], -src.normals[1] };

		std::vector<uint8_t> nglb;
		if( !t2glb::mesh_to_projected_glb( tv, 4, tt, 2, sv, 4, st, 2, spbr, nopt, nglb, err ) ) {
			std::fprintf( stderr, "normal map bake failed: %s\n", err.c_str() );
			return 1;
		}
		std::string			 njson;
		const uint8_t*		 nbin = nullptr;
		std::vector<GlbView> nviews;
		if( !glb_open( nglb, njson, nbin, nviews ) || nviews.size() != 8 ) {
			std::fprintf( stderr, "normal map GLB has %zu bufferViews, expected 8\n", nviews.size() );
			return 1;
		}
		if( njson.find( "\"TANGENT\"" ) == std::string::npos || njson.find( "\"normalTexture\"" ) == std::string::npos ) {
			std::fprintf( stderr, "normal map GLB is missing TANGENT or normalTexture\n" );
			return 1;
		}

		const float*   gnrm = ( const float* )( nbin + nviews[1].off );
		const float*   gtan = ( const float* )( nbin + nviews[3].off );
		const uint32_t gnv	= nviews[1].len / ( 3 * sizeof( float ) );
		if( gnv < 3 || nviews[3].len != gnv * 4 * sizeof( float ) ) {
			std::fprintf( stderr, "TANGENT does not cover every vertex (%u verts)\n", gnv );
			return 1;
		}
		// One planar chart: the frame must be the same at every vertex, or the
		// single-frame reconstruction below would not be valid.
		for( uint32_t v = 1; v < gnv; ++v )
			for( int k = 0; k < 4; ++k ) {
				if( k < 3 && std::fabs( gnrm[3 * v + k] - gnrm[k] ) > 1e-4f ) {
					std::fprintf( stderr, "planar target has a varying NORMAL at vertex %u\n", v );
					return 1;
				}
				if( std::fabs( gtan[4 * v + k] - gtan[k] ) > 1e-4f ) {
					std::fprintf( stderr, "planar target has a varying TANGENT at vertex %u\n", v );
					return 1;
				}
			}
		const float	   N[3] = { gnrm[0], gnrm[1], gnrm[2] };
		const float	   T[3] = { gtan[0], gtan[1], gtan[2] };
		const float	   w	= gtan[3] < 0.0f ? -1.0f : 1.0f;
		const float	   B[3] = { ( N[1] * T[2] - N[2] * T[1] ) * w, ( N[2] * T[0] - N[0] * T[2] ) * w, ( N[0] * T[1] - N[1] * T[0] ) * w };

		int			   iw = 0, ih = 0, ic = 0;
		unsigned char* px = stbi_load_from_memory( nbin + nviews[7].off, ( int )nviews[7].len, &iw, &ih, &ic, 3 );
		if( !px || iw <= 0 || ih <= 0 ) {
			std::fprintf( stderr, "could not decode the baked normal map\n" );
			return 1;
		}
		double worst_deg = 0.0;
		int	   sampled	 = 0;
		for( int i = 0; i < iw * ih; ++i ) {
			const unsigned char* t = px + ( size_t )i * 3;
			if( t[0] == 128 && t[1] == 128 && t[2] == 255 )
				continue; // untouched gutter / deliberately flat
			float n[3];
			for( int k = 0; k < 3; ++k )
				n[k] = ( float )t[k] / 255.0f * 2.0f - 1.0f;
			float world[3];
			for( int k = 0; k < 3; ++k )
				world[k] = n[0] * T[k] + n[1] * B[k] + n[2] * N[k];
			const float l = std::sqrt( world[0] * world[0] + world[1] * world[1] + world[2] * world[2] );
			if( l < 1e-6f )
				continue;
			double dot = 0.0;
			for( int k = 0; k < 3; ++k )
				dot += ( double )world[k] / l * want[k];
			dot		  = dot > 1.0 ? 1.0 : ( dot < -1.0 ? -1.0 : dot );
			worst_deg = std::fmax( worst_deg, std::acos( dot ) * 180.0 / 3.14159265358979323846 );
			++sampled;
		}
		stbi_image_free( px );

		const t2glb::BakeTimings bt = t2glb::last_bake_timings();
		std::printf( "normal map round trip: %d of %d texels reconstructed, worst %.3f deg off the source normal; %d of %d covered texels rejected\n",
			sampled,
			iw * ih,
			worst_deg,
			bt.normal_rejected,
			bt.normal_texels );
		// A vacuous pass would be the real failure here: if nothing was
		// reconstructed the bake wrote a uniformly flat map and the check below
		// would be meaningless.
		if( sampled < iw * ih / 8 ) {
			std::fprintf( stderr, "the baked normal map is essentially flat (%d of %d texels carry a direction)\n", sampled, iw * ih );
			return 1;
		}
		// 8-bit quantization alone costs ~0.25 deg per axis; the rest is the
		// dilation pass averaging across the chart border.
		if( worst_deg > 2.0 ) {
			std::fprintf( stderr, "baked normals reconstruct %.3f deg away from the source normal\n", worst_deg );
			return 1;
		}
		if( bt.normal_rejected != 0 ) {
			std::fprintf( stderr, "%d texels were rejected on a source that faces the target everywhere\n", bt.normal_rejected );
			return 1;
		}

		// The escape hatch has to actually disable the feature.
		nopt.normal_map = false;
		std::vector<uint8_t> plain;
		if( !t2glb::mesh_to_projected_glb( tv, 4, tt, 2, sv, 4, st, 2, spbr, nopt, plain, err ) ) {
			std::fprintf( stderr, "bake without the normal map failed: %s\n", err.c_str() );
			return 1;
		}
		const std::string plain_json( ( const char* )plain.data(), plain.size() );
		if( plain_json.find( "\"normalTexture\"" ) != std::string::npos || plain_json.find( "\"TANGENT\"" ) != std::string::npos ) {
			std::fprintf( stderr, "normal_map = false still emitted a normal map\n" );
			return 1;
		}
	}

	std::puts( "RESULT: PASS" );
	return 0;
}
