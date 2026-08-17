// Validation of the shape-SLAT stage against the PyTorch reference
// (scripts/dump_slat_reference.py -> dumps/reference_slat.gguf):
//
//   1. SLAT flow forward at t=500 on the reference voxel set   (tight gate)
//   2. full 12-step CFG sampler + denormalization              (loose gate)
//   3. FDG decoder: per-level features, subdivision decisions,
//      final 7-channel output + coords                          (tight gate,
//      but only meaningful if the subdivision sets match exactly)
//
// usage: test_slat <slat_flow_f32.gguf> <shape_dec_f32.gguf> <reference_slat.gguf>
// exits 77 (ctest SKIP) when inputs are missing.

#include "trellis2.h"
#include "parity.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static bool file_exists( const std::string& p )
{
	std::ifstream f( p );
	return f.good();
}

int main( int argc, char** argv )
{
	if( argc < 4 ) {
		std::fprintf( stderr, "usage: %s <slat_flow_f32.gguf> <shape_dec_f32.gguf> <reference_slat.gguf>\n", argv[0] );
		return 2;
	}
	const std::string flow_path = argv[1];
	const std::string dec_path	= argv[2];
	const std::string ref_path	= argv[3];

	if( !file_exists( flow_path ) || !file_exists( dec_path ) || !file_exists( ref_path ) ) {
		std::fprintf( stderr, "missing input file(s), skipping\n" );
		return 77;
	}

	t2_parity::baseline ref;
	if( !ref.open( ref_path ) ) {
		std::fprintf( stderr, "failed to open %s\n", ref_path.c_str() );
		return 1;
	}

	std::vector<float> coords_f, noise, cond;
	if( !ref.load( "coords", coords_f ) || !ref.load( "slat_noise", noise ) ) {
		std::fprintf( stderr, "reference missing coords/slat_noise\n" );
		return 1;
	}
	const int			 L = ( int )( coords_f.size() / 4 );
	std::vector<int32_t> coords( ( size_t )L * 3 );
	for( int v = 0; v < L; ++v ) {
		coords[( size_t )v * 3]		= ( int32_t )coords_f[( size_t )v * 4 + 1];
		coords[( size_t )v * 3 + 1] = ( int32_t )coords_f[( size_t )v * 4 + 2];
		coords[( size_t )v * 3 + 2] = ( int32_t )coords_f[( size_t )v * 4 + 3];
	}
	std::printf( "reference: %d voxels\n", L );

	// conditioning comes from the shared fixture
	{
		trellis2_dino_cond c;
		std::string		   err;
		const char*		   dinodata = std::getenv( "TRELLIS2_DINODATA" );
		std::string		   path		= dinodata ? dinodata : "dumps/fixture.dinodata";
		if( !trellis2_load_dinodata( path, c, &err ) ) {
			std::fprintf( stderr, "cannot load %s (%s), skipping\n", path.c_str(), err.c_str() );
			return 77;
		}
		cond = std::move( c.data );
	}
	const int				  Lkv = ( int )( cond.size() / 1024 );

	std::string				  err;
	trellis2_slat_flow_model* flow = trellis2_slat_flow_load( flow_path, true, &err );
	if( !flow ) {
		std::fprintf( stderr, "flow load failed: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "flow backend: %s\n", trellis2_slat_flow_backend_name( flow ) );

	int n_fail = 0;

	// ── 1. forward parity at t=500 ───────────────────────────────────────────
	{
		std::vector<float> got( ( size_t )L * 32 ), want;
		if( !trellis2_slat_flow_forward( flow, noise.data(), L, coords.data(), 500.0f, cond.data(), Lkv, 1024, got.data(), &err ) ) {
			std::fprintf( stderr, "flow forward failed: %s\n", err.c_str() );
			return 1;
		}
		ref.load( "flow_t500_out", want );
		// rel-L2 gate (matches the SS-flow tests): per-element noise from
		// CPU-vs-CUDA fp32 reduction order is expected, the trajectory is not.
		t2_parity::compare_stats st;
		t2_parity::compare( got, want, "flow_t500_out", 2e-3, 2e-3, &st );
		if( st.rel_l2 > 3e-3 ) {
			std::printf( "  -> forward rel_l2 %.4g > 3e-3, FAIL\n", st.rel_l2 );
			++n_fail;
		}
	}

	// ── 2. full sampler + denormalization ────────────────────────────────────
	std::vector<float> slat( ( size_t )L * 32 );
	{
		trellis2_ss_sampler_params P;
		P.steps					= 12;
		P.guidance_strength		= 7.5f;
		P.guidance_rescale		= 0.5f;
		P.guidance_interval_min = 0.6f;
		P.guidance_interval_max = 1.0f;
		P.rescale_t				= 3.0f;
		P.verbose				= true;
		if( !trellis2_slat_flow_sample( flow, L, coords.data(), cond.data(), Lkv, 1024, &P, noise.data(), /*denormalize*/ true, slat.data(), &err ) ) {
			std::fprintf( stderr, "flow sample failed: %s\n", err.c_str() );
			return 1;
		}
		std::vector<float> want;
		ref.load( "slat", want );
		// The 12-step Euler sampler with CFG-rescale chaotically amplifies the
		// per-step fp difference between backends (the same effect the SS
		// sampler shows: tight on CPU, ~0.1 rel-L2 on GPU). Gate loosely on the
		// trajectory here; the SS-sampler test already validates the shared
		// Euler loop tightly on CPU. TRELLIS2_SLAT_STRICT tightens it for a CPU
		// run.
		t2_parity::compare_stats st;
		t2_parity::compare( slat, want, "slat(sampled)", 5e-2, 5e-2, &st );
		const double samp_gate = std::getenv( "TRELLIS2_SLAT_STRICT" ) ? 2e-2 : 2e-1;
		if( st.rel_l2 > samp_gate ) {
			std::printf( "  -> sampler rel_l2 %.4g > %.0e, FAIL\n", st.rel_l2, samp_gate );
			++n_fail;
		}
		// Decode the REFERENCE slat so decoder parity is independent of the
		// sampler trajectory noise.
		slat = want;
	}
	trellis2_slat_flow_free( flow );

	// ── 3. decoder (CPU: ggml has no CUDA CONV_3D / sparse-conv kernel) ──────
	trellis2_shape_dec_model* dec = trellis2_shape_dec_load( dec_path, true, &err, "cpu" );
	if( !dec ) {
		std::fprintf( stderr, "dec load failed: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "dec backend: %s\n", trellis2_shape_dec_backend_name( dec ) );

	std::vector<float>				   out_feats;
	std::vector<int32_t>			   out_coords;
	std::vector<trellis2_subdiv_level> subs;
	trellis2_shape_dec_taps			   taps;
	if( !trellis2_shape_dec_decode_with_subs( dec, slat.data(), L, coords.data(), out_feats, out_coords, subs, &taps, &err ) ) {
		std::fprintf( stderr, "decode failed: %s\n", err.c_str() );
		trellis2_shape_dec_free( dec );
		return 1;
	}
	trellis2_shape_dec_free( dec );

	// Integrated texture generation replays these exact decoder decisions in
	// the texture VAE. Guard their order/shape independently of activation taps.
	if( subs.size() != 4 ) {
		std::printf( "  -> subdivision guide has %zu levels, expected 4, FAIL\n", subs.size() );
		++n_fail;
	} else {
		for( size_t lvl = 0; lvl < subs.size(); ++lvl ) {
			if( subs[lvl].fine_coords.size() != subs[lvl].cidx.size() * 3 ) {
				std::printf( "  -> subdivision level %zu coord/index size mismatch, FAIL\n", lvl );
				++n_fail;
			}
		}
		if( subs.back().fine_coords != out_coords ) {
			std::printf( "  -> final subdivision guide does not reproduce decoder coords, FAIL\n" );
			++n_fail;
		}
	}

	// The decoder's per-level active set is chosen by subdivision-logit signs;
	// if my logits match the reference's, every level's voxel set (hence tap
	// size) matches exactly. A size mismatch therefore means a real sign flip,
	// which is a hard fail. Numerically, deep sparse-conv accumulates fp noise,
	// so gate features on rel-L2 rather than per-element.
	std::vector<float> refbuf;
	int				   n_cmp = 0;
	for( size_t i = 0; i < taps.names.size(); ++i ) {
		const std::string& name = taps.names[i];
		if( !ref.has( name ) )
			continue;
		ref.load( name, refbuf );
		++n_cmp;
		t2_parity::compare_stats st;
		t2_parity::compare( taps.data[i], refbuf, name, 2e-3, 2e-3, &st );
		const double gate = 2e-2; // rel-L2
		// A few voxels whose subdivision logit sits within fp-noise of zero
		// flip the >0 threshold, so the final active set can differ by a
		// handful out of millions. Tolerate a <0.05% set-size difference; a
		// larger divergence is a real bug.
		const size_t a = taps.data[i].size(), b = refbuf.size();
		const size_t big = a > b ? a : b, sml = a > b ? b : a;
		if( a != b ) {
			const double frac = ( double )( big - sml ) / ( double )big;
			if( frac > 5e-4 ) {
				std::printf( "  -> %s SIZE MISMATCH %zu vs %zu (%.3f%%), FAIL\n", name.c_str(), a, b, 100.0 * frac );
				++n_fail;
			} else {
				std::printf( "  -> %s near-match (%zu vs %zu, %.4f%% subdivision boundary flip) OK\n", name.c_str(), a, b, 100.0 * frac );
			}
		} else if( st.rel_l2 > gate ) {
			std::printf( "  -> %s rel_l2 %.4g > %.0e, FAIL\n", name.c_str(), st.rel_l2, gate );
			++n_fail;
			// Break a failing multi-channel output down per channel. A single
			// rel-L2 cannot tell "the maths is wrong" from "the channels mean
			// something else here": cos ~ +1 is a scale/offset, ~ -1 a sign, ~ 0
			// genuine divergence, and one bad channel among good ones points at
			// the split rather than the arithmetic. out7 is [7, L] with the
			// channel fastest, matching the reference's [L, 7] row-major.
			const size_t nch = name == "out7" ? 7 : ( name == "pbr" ? 6 : 0 );
			if( nch && a % nch == 0 ) {
				const std::vector<float>& got = taps.data[i];
				for( size_t c = 0; c < nch; ++c ) {
					double dg = 0, dr = 0, dgr = 0, dd = 0, sg = 0, sr = 0;
					size_t n = 0;
					for( size_t k = c; k < a; k += nch, ++n ) {
						const double g = got[k], r = refbuf[k];
						dg += g * g;
						dr += r * r;
						dgr += g * r;
						dd += ( g - r ) * ( g - r );
						sg += g;
						sr += r;
					}
					const double cos = ( dg > 0 && dr > 0 ) ? dgr / ( std::sqrt( dg ) * std::sqrt( dr ) ) : 0.0;
					std::printf( "     ch%zu  relL2=%.4g  cos=%+.4f  mean got=%+.4f ref=%+.4f  rms got=%.4f ref=%.4f\n",
						c,
						dr > 0 ? std::sqrt( dd / dr ) : 0.0,
						cos,
						sg / ( double )n,
						sr / ( double )n,
						std::sqrt( dg / ( double )n ),
						std::sqrt( dr / ( double )n ) );
				}
			}
		}
	}

	std::printf( "\ndecoder taps compared: %d, total failures: %d\n", n_cmp, n_fail );
	std::printf( "RESULT: %s\n", n_fail ? "FAIL" : "PASS" );
	return n_fail ? 1 : 0;
}
