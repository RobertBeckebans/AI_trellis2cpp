// Validation of the PBR-texture stages against the PyTorch reference
// (scripts/dump_texture_reference.py -> dumps/reference_texture.gguf):
//
//   1. shape encoder: dual grid -> shape SLAT (32ch)          (coord-matched)
//   2. tex flow forward at t=500 with concat_cond             (tight gate)
//   3. tex flow full sampler + denorm                         (loose gate)
//   4. tex decoder: tex SLAT -> 6ch PBR, replaying the         (coord-matched)
//      encoder's subdivision
//
// usage: test_texture <shape_enc.gguf> <tex_flow_512.gguf> <tex_dec.gguf> <reference_texture.gguf>
// exits 77 (ctest SKIP) when inputs are missing.

#include "trellis2.h"
#include "parity.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static bool file_exists( const std::string& p )
{
	std::ifstream f( p );
	return f.good();
}

static uint64_t vkey( int32_t x, int32_t y, int32_t z )
{
	return ( ( uint64_t )( uint32_t )x << 42 ) | ( ( uint64_t )( uint32_t )y << 21 ) | ( uint64_t )( uint32_t )z;
}

// coords4 is the reference [N,4] (batch,x,y,z) dump; build key->row.
static std::unordered_map<uint64_t, int> coord_map( const std::vector<float>& c4 )
{
	std::unordered_map<uint64_t, int> m;
	const int						  n = ( int )( c4.size() / 4 );
	m.reserve( ( size_t )n * 2 );
	for( int v = 0; v < n; ++v )
		m[vkey( ( int32_t )c4[( size_t )v * 4 + 1], ( int32_t )c4[( size_t )v * 4 + 2], ( int32_t )c4[( size_t )v * 4 + 3] )] = v;
	return m;
}

// Reorder `got` (rows in `got_coords3` order) into the reference row order given
// by `ref_coords4`, returning the aligned buffer; reports set mismatches.
static bool align_to_ref( const std::vector<float>& got, const std::vector<int32_t>& got_coords3, const std::vector<float>& ref_coords4, int ch, std::vector<float>& out, const char* label )
{
	const int ng = ( int )( got_coords3.size() / 3 );
	const int nr = ( int )( ref_coords4.size() / 4 );
	if( ng != nr ) {
		std::printf( "[%-16s] VOXEL COUNT got=%d ref=%d -> FAIL\n", label, ng, nr );
		return false;
	}
	std::unordered_map<uint64_t, int> gm;
	gm.reserve( ( size_t )ng * 2 );
	for( int v = 0; v < ng; ++v )
		gm[vkey( got_coords3[( size_t )v * 3], got_coords3[( size_t )v * 3 + 1], got_coords3[( size_t )v * 3 + 2] )] = v;
	out.resize( ( size_t )nr * ch );
	int miss = 0;
	for( int r = 0; r < nr; ++r ) {
		auto it = gm.find( vkey( ( int32_t )ref_coords4[( size_t )r * 4 + 1], ( int32_t )ref_coords4[( size_t )r * 4 + 2], ( int32_t )ref_coords4[( size_t )r * 4 + 3] ) );
		if( it == gm.end() ) {
			++miss;
			continue;
		}
		std::memcpy( out.data() + ( size_t )r * ch, got.data() + ( size_t )it->second * ch, ( size_t )ch * sizeof( float ) );
	}
	if( miss ) {
		std::printf( "[%-16s] %d/%d ref voxels absent in got -> FAIL\n", label, miss, nr );
		return false;
	}
	return true;
}

int main( int argc, char** argv )
{
	if( argc < 5 ) {
		std::fprintf( stderr, "usage: %s <shape_enc.gguf> <tex_flow_512.gguf> <tex_dec.gguf> <reference_texture.gguf>\n", argv[0] );
		return 2;
	}
	const std::string enc_path = argv[1], flow_path = argv[2], dec_path = argv[3], ref_path = argv[4];
	if( !file_exists( enc_path ) || !file_exists( flow_path ) || !file_exists( dec_path ) || !file_exists( ref_path ) ) {
		std::fprintf( stderr, "missing input file(s), skipping\n" );
		return 77;
	}

	t2_parity::baseline ref;
	if( !ref.open( ref_path ) ) {
		std::fprintf( stderr, "failed to open %s\n", ref_path.c_str() );
		return 1;
	}

	std::vector<float> enc_vert, enc_inter, enc_coords4, cond;
	std::vector<float> shape_slat_ref, shape_coords4, tex_noise, tex_slat_ref, pbr_ref, pbr_coords4;
	if( !ref.load( "enc_vert", enc_vert ) || !ref.load( "enc_inter", enc_inter ) || !ref.load( "enc_coords", enc_coords4 ) || !ref.load( "cond", cond ) || !ref.load( "shape_slat", shape_slat_ref ) ||
		!ref.load( "shape_coords", shape_coords4 ) || !ref.load( "tex_noise", tex_noise ) || !ref.load( "tex_slat", tex_slat_ref ) || !ref.load( "pbr", pbr_ref ) ||
		!ref.load( "pbr_coords", pbr_coords4 ) ) {
		std::fprintf( stderr, "reference missing a required tensor\n" );
		return 1;
	}
	const int N	  = ( int )( enc_coords4.size() / 4 );	 // fine voxels (encoder input)
	const int Nl  = ( int )( shape_coords4.size() / 4 ); // latent voxels
	const int Lkv = ( int )( cond.size() / 1024 );
	std::printf( "reference: %d dual-grid voxels, %d latent voxels, %d cond tokens\n", N, Nl, Lkv );

	std::vector<int32_t> enc_coords( ( size_t )N * 3 );
	std::vector<float>	 in6( ( size_t )N * 6 );
	for( int v = 0; v < N; ++v ) {
		enc_coords[( size_t )v * 3]		= ( int32_t )enc_coords4[( size_t )v * 4 + 1];
		enc_coords[( size_t )v * 3 + 1] = ( int32_t )enc_coords4[( size_t )v * 4 + 2];
		enc_coords[( size_t )v * 3 + 2] = ( int32_t )enc_coords4[( size_t )v * 4 + 3];
		for( int c = 0; c < 3; ++c ) {
			in6[( size_t )v * 6 + c]	 = enc_vert[( size_t )v * 3 + c];
			in6[( size_t )v * 6 + 3 + c] = enc_inter[( size_t )v * 3 + c];
		}
	}
	std::vector<int32_t> lat_coords( ( size_t )Nl * 3 );
	for( int v = 0; v < Nl; ++v ) {
		lat_coords[( size_t )v * 3]		= ( int32_t )shape_coords4[( size_t )v * 4 + 1];
		lat_coords[( size_t )v * 3 + 1] = ( int32_t )shape_coords4[( size_t )v * 4 + 2];
		lat_coords[( size_t )v * 3 + 2] = ( int32_t )shape_coords4[( size_t )v * 4 + 3];
	}

	std::string				  err;
	int						  n_fail = 0;

	// ── 1. shape encoder ─────────────────────────────────────────────────────
	trellis2_shape_enc_model* enc = trellis2_shape_enc_load( enc_path, true, &err );
	if( !enc ) {
		std::fprintf( stderr, "enc load failed: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "enc backend: %s\n", trellis2_shape_enc_backend_name( enc ) );

	std::vector<float>				   shape_slat;
	std::vector<int32_t>			   shape_coords;
	std::vector<trellis2_subdiv_level> subs;
	if( !trellis2_shape_enc_encode( enc, in6.data(), N, enc_coords.data(), shape_slat, shape_coords, subs, nullptr, &err ) ) {
		std::fprintf( stderr, "encode failed: %s\n", err.c_str() );
		trellis2_shape_enc_free( enc );
		return 1;
	}
	trellis2_shape_enc_free( enc );
	std::printf( "encoder out: %zu latent voxels\n", shape_coords.size() / 3 );
	{
		std::vector<float> aligned;
		if( !align_to_ref( shape_slat, shape_coords, shape_coords4, 32, aligned, "shape_slat" ) ) {
			++n_fail;
		} else {
			t2_parity::compare_stats st;
			t2_parity::compare( aligned, shape_slat_ref, "shape_slat", 5e-3, 5e-3, &st );
			if( st.rel_l2 > 3e-2 ) {
				std::printf( "  -> shape_slat rel_l2 %.4g > 3e-2, FAIL\n", st.rel_l2 );
				++n_fail;
			}
		}
	}

	// ── 2 & 3. tex flow (concat_cond) ────────────────────────────────────────
	trellis2_slat_flow_model* flow = trellis2_slat_flow_load( flow_path, true, &err );
	if( !flow ) {
		std::fprintf( stderr, "flow load failed: %s\n", err.c_str() );
		return 1;
	}
	const trellis2_slat_flow_hparams& fhp = trellis2_slat_flow_hparams_of( flow );
	std::printf( "flow backend: %s (in=%d out=%d concat=%d)\n", trellis2_slat_flow_backend_name( flow ), fhp.in_channels, fhp.out_channels, fhp.concat_cond_channels );

	// 2. forward @ t=500: build [noise(32) | normalized shape_slat(32)] manually.
	{
		std::vector<float> xin( ( size_t )Nl * fhp.in_channels ), got( ( size_t )Nl * 32 ), want;
		for( int v = 0; v < Nl; ++v ) {
			for( int c = 0; c < 32; ++c )
				xin[( size_t )v * 64 + c] = tex_noise[( size_t )v * 32 + c];
			for( int c = 0; c < 32; ++c )
				xin[( size_t )v * 64 + 32 + c] = ( shape_slat_ref[( size_t )v * 32 + c] - fhp.concat_norm_mean[c] ) / fhp.concat_norm_std[c];
		}
		if( !trellis2_slat_flow_forward( flow, xin.data(), Nl, lat_coords.data(), 500.0f, cond.data(), Lkv, 1024, got.data(), &err ) ) {
			std::fprintf( stderr, "flow forward failed: %s\n", err.c_str() );
			return 1;
		}
		ref.load( "tex_flow_t500", want );
		t2_parity::compare_stats st;
		t2_parity::compare( got, want, "tex_flow_t500", 2e-3, 2e-3, &st );
		if( st.rel_l2 > 5e-3 ) {
			std::printf( "  -> forward rel_l2 %.4g > 5e-3, FAIL\n", st.rel_l2 );
			++n_fail;
		}
	}

	// 3. full tex sampler.
	{
		// Texture sampler params (texturing_pipeline.json tex_slat_sampler):
		// guidance_strength 1.0 (no CFG amplification), rescale 0, interval [0.6,0.9].
		trellis2_ss_sampler_params P;
		P.steps					= 12;
		P.guidance_strength		= 1.0f;
		P.guidance_rescale		= 0.0f;
		P.guidance_interval_min = 0.6f;
		P.guidance_interval_max = 0.9f;
		P.rescale_t				= 3.0f;
		P.verbose				= true;
		std::vector<float> got( ( size_t )Nl * 32 ), want;
		if( !trellis2_slat_flow_sample_tex( flow,
				Nl,
				lat_coords.data(),
				cond.data(),
				Lkv,
				1024,
				shape_slat_ref.data(),
				&P,
				tex_noise.data(),
				/*denormalize*/ true,
				got.data(),
				&err ) ) {
			std::fprintf( stderr, "tex sample failed: %s\n", err.c_str() );
			return 1;
		}
		ref.load( "tex_slat", want );
		t2_parity::compare_stats st;
		t2_parity::compare( got, want, "tex_slat(sampled)", 5e-2, 5e-2, &st );
		const double gate = std::getenv( "TRELLIS2_SLAT_STRICT" ) ? 2e-2 : 2e-1;
		if( st.rel_l2 > gate ) {
			std::printf( "  -> tex sampler rel_l2 %.4g > %.0e, FAIL\n", st.rel_l2, gate );
			++n_fail;
		}
	}
	trellis2_slat_flow_free( flow );

	// ── 4. tex decoder: decode the REFERENCE tex SLAT (isolate from sampler) ──
	// Align the reference tex SLAT into the encoder's latent order, then decode
	// with the encoder's subdivisions.
	std::vector<float> tex_slat_aligned;
	if( !align_to_ref( tex_slat_ref, lat_coords, shape_coords4, 32, tex_slat_aligned, "tex_slat_align" ) ) {
		// reference latent order should already match ours; if not, fail loudly
		std::printf( "  -> could not align tex SLAT to encoder latent order, FAIL\n" );
		return ++n_fail, ( n_fail ? 1 : 0 );
	}
	// aligned is in reference order; but the decoder wants it in the encoder's
	// order (shape_coords). Reorder reference->encoder via the encoder coords.
	std::vector<float> tex_slat_enc( ( size_t )Nl * 32 );
	{
		std::unordered_map<uint64_t, int> rm = coord_map( shape_coords4 );
		for( int v = 0; v < Nl; ++v ) {
			auto	  it = rm.find( vkey( shape_coords[( size_t )v * 3], shape_coords[( size_t )v * 3 + 1], shape_coords[( size_t )v * 3 + 2] ) );
			const int rr = ( it == rm.end() ) ? v : it->second;
			std::memcpy( tex_slat_enc.data() + ( size_t )v * 32, tex_slat_ref.data() + ( size_t )rr * 32, 32 * sizeof( float ) );
		}
	}

	trellis2_shape_dec_model* texdec = trellis2_tex_dec_load( dec_path, true, &err );
	if( !texdec ) {
		std::fprintf( stderr, "tex_dec load failed: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "tex_dec backend: %s\n", trellis2_shape_dec_backend_name( texdec ) );

	std::vector<float>	 pbr;
	std::vector<int32_t> pbr_coords;
	if( !trellis2_tex_dec_decode( texdec, tex_slat_enc.data(), Nl, shape_coords.data(), subs, pbr, pbr_coords, &err ) ) {
		std::fprintf( stderr, "tex decode failed: %s\n", err.c_str() );
		trellis2_shape_dec_free( texdec );
		return 1;
	}
	trellis2_shape_dec_free( texdec );
	std::printf( "tex decode out: %zu PBR voxels\n", pbr_coords.size() / 3 );
	{
		std::vector<float> aligned;
		if( !align_to_ref( pbr, pbr_coords, pbr_coords4, 6, aligned, "pbr" ) ) {
			++n_fail;
		} else {
			t2_parity::compare_stats st;
			t2_parity::compare( aligned, pbr_ref, "pbr", 5e-3, 5e-3, &st );
			if( st.rel_l2 > 3e-2 ) {
				std::printf( "  -> pbr rel_l2 %.4g > 3e-2, FAIL\n", st.rel_l2 );
				++n_fail;
			}
		}
	}

	std::printf( "\nRESULT: %s (failures: %d)\n", n_fail ? "FAIL" : "PASS", n_fail );
	return n_fail ? 1 : 0;
}
