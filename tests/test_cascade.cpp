// Validation of the cascade HR stage against the PyTorch reference
// (scripts/dump_cascade_reference.py -> dumps/reference_cascade.gguf):
//
//   1. shape-decoder upsample(x4): the LR slat -> 512^3 candidate coord set,
//      and its quantized 64^3 HR scaffold (subdivision-boundary set tolerance).
//   1b. the same candidate set at the 1536 tier: the token budget's choice and
//      the resulting 96^3 scaffold, against the reference when the dump carries
//      one (hr_coords_1536), otherwise as invariants.
//   2. HR (1024-model) flow forward at t=500 on the HR scaffold  (tight gate).
//   3. final 1024^3 decode of the reference HR slat: per-level features,
//      subdivision, and 7-channel output.
//   (The 12-step HR sampler is env-gated (TRELLIS2_CASCADE_SAMPLE) — 24 forwards
//    at ~40k tokens is impractical on CPU; the shared Euler loop is already
//    validated by test_slat / test_ss_sample.)
//
// Sections 2 and 3 need tensors that a dump written with
// scripts/dump_cascade_reference.py --skip-hr-sampler does not contain. They
// then report "not in the dump" and are skipped; sections 1 and 1b, which are
// what the 1536 tier is gated on, run either way. Reporting a deliberately
// absent tensor is the correct outcome — the test still gates everything the
// dump does carry, rather than exiting 77.
//
// usage: test_cascade <slat_512_f32> <slat_1024_f32> <shape_dec_f32> <reference_cascade.gguf>
// exits 77 (ctest SKIP) when inputs are missing.

#include "trellis2.h"
#include "cascade_tokens.h"
#include "parity.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

static bool file_exists( const std::string& p )
{
	std::ifstream f( p );
	return f.good();
}

// [L*4] (batch,x,y,z) reference coords -> [L*3] int32 (x,y,z)
static std::vector<int32_t> coords_xyz( const std::vector<float>& c4 )
{
	const size_t		 L = c4.size() / 4;
	std::vector<int32_t> out( L * 3 );
	for( size_t v = 0; v < L; ++v ) {
		out[v * 3]	   = ( int32_t )c4[v * 4 + 1];
		out[v * 3 + 1] = ( int32_t )c4[v * 4 + 2];
		out[v * 3 + 2] = ( int32_t )c4[v * 4 + 3];
	}
	return out;
}

static uint64_t vkey( int32_t x, int32_t y, int32_t z )
{
	return ( ( uint64_t )( uint32_t )x << 40 ) | ( ( uint64_t )( uint32_t )y << 20 ) | ( uint64_t )( uint32_t )z;
}

// symmetric-difference fraction of two coord sets ([*3] int32)
static double set_diff_frac( const std::vector<int32_t>& a, const std::vector<int32_t>& b )
{
	std::unordered_set<uint64_t> sa, sb;
	for( size_t i = 0; i < a.size(); i += 3 )
		sa.insert( vkey( a[i], a[i + 1], a[i + 2] ) );
	for( size_t i = 0; i < b.size(); i += 3 )
		sb.insert( vkey( b[i], b[i + 1], b[i + 2] ) );
	size_t only = 0;
	for( uint64_t k : sa )
		if( !sb.count( k ) )
			++only;
	for( uint64_t k : sb )
		if( !sa.count( k ) )
			++only;
	const size_t big = sa.size() > sb.size() ? sa.size() : sb.size();
	return big ? ( double )only / ( double )big : 0.0;
}

int main( int argc, char** argv )
{
	if( argc < 5 ) {
		std::fprintf( stderr, "usage: %s <slat_512_f32.gguf> <slat_1024_f32.gguf> <shape_dec_f32.gguf> <reference_cascade.gguf>\n", argv[0] );
		return 2;
	}
	const std::string flow512_path	= argv[1];
	const std::string flow1024_path = argv[2];
	const std::string dec_path		= argv[3];
	const std::string ref_path		= argv[4];
	if( !file_exists( flow512_path ) || !file_exists( flow1024_path ) || !file_exists( dec_path ) || !file_exists( ref_path ) ) {
		std::fprintf( stderr, "missing input file(s), skipping\n" );
		return 77;
	}

	t2_parity::baseline ref;
	if( !ref.open( ref_path ) ) {
		std::fprintf( stderr, "failed to open %s\n", ref_path.c_str() );
		return 1;
	}

	std::vector<float> cond512, cond1024, coords32f, lr_slat, up_coordsf, hr_coordsf, hr_noise;
	if( !ref.load( "cond_512", cond512 ) || !ref.load( "cond_1024", cond1024 ) || !ref.load( "coords32", coords32f ) || !ref.load( "lr_slat", lr_slat ) || !ref.load( "up_coords", up_coordsf ) ||
		!ref.load( "hr_coords", hr_coordsf ) ) {
		std::fprintf( stderr, "reference missing required tensors\n" );
		return 1;
	}
	// hr_noise, and everything derived from it (hr_flow_t500_out, hr_slat, out7),
	// is absent from a dump written with --skip-hr-sampler, which captures only
	// the coordinate chain. That is a deliberately partial dump, not a broken one:
	// section 1 — the part the 1536 tier is gated on — runs either way, and
	// sections 2 and 3 report the tensors as not present instead of failing.
	const bool			 have_hr_chain = ref.load( "hr_noise", hr_noise );
	const int			 L32		   = ( int )( coords32f.size() / 4 );
	const int			 Lhr		   = ( int )( hr_coordsf.size() / 4 );
	const int			 Lkv512		   = ( int )( cond512.size() / 1024 );
	const int			 Lkv1024	   = ( int )( cond1024.size() / 1024 );
	std::vector<int32_t> coords32	   = coords_xyz( coords32f );
	std::vector<int32_t> ref_up_coords = coords_xyz( up_coordsf );
	std::vector<int32_t> hr_coords	   = coords_xyz( hr_coordsf );
	std::printf( "reference: scaffold %d voxels, upsample %zu, HR %d voxels, cond %d/%d tokens\n", L32, up_coordsf.size() / 4, Lhr, Lkv512, Lkv1024 );

	std::string				  err;
	int						  n_fail = 0;

	// ── 1. shape-decoder upsample(x4): coord set + quantized 64^3 scaffold ────
	trellis2_shape_dec_model* dec = trellis2_shape_dec_load( dec_path, true, &err, "cpu" );
	if( !dec ) {
		std::fprintf( stderr, "dec load failed: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "dec backend: %s\n", trellis2_shape_dec_backend_name( dec ) );
	{
		std::vector<int32_t> got_up;
		if( !trellis2_shape_dec_upsample( dec,
				lr_slat.data(),
				L32,
				coords32.data(),
				/*upsample_times*/ 4,
				got_up,
				&err ) ) {
			std::fprintf( stderr, "upsample failed: %s\n", err.c_str() );
			trellis2_shape_dec_free( dec );
			return 1;
		}
		const double up_frac = set_diff_frac( got_up, ref_up_coords );
		std::printf( "[upsample coords] got %zu vs ref %zu, sym-diff %.4f%% -> %s\n", got_up.size() / 3, ref_up_coords.size() / 3, 100.0 * up_frac, up_frac <= 5e-4 ? "OK" : "FAIL" );
		if( up_frac > 5e-4 )
			++n_fail;

		// quantize both to 64^3 and compare the dedup'd set that drives the HR flow
		auto quant = []( const std::vector<int32_t>& c ) {
			std::unordered_set<uint64_t> s;
			std::vector<int32_t>		 out;
			for( size_t i = 0; i < c.size(); i += 3 ) {
				int32_t x = ( int32_t )( ( c[i] + 0.5f ) / 512.0f * 64.0f );
				int32_t y = ( int32_t )( ( c[i + 1] + 0.5f ) / 512.0f * 64.0f );
				int32_t z = ( int32_t )( ( c[i + 2] + 0.5f ) / 512.0f * 64.0f );
				if( s.insert( vkey( x, y, z ) ).second ) {
					out.push_back( x );
					out.push_back( y );
					out.push_back( z );
				}
			}
			return out;
		};
		std::vector<int32_t> my_hr	 = quant( got_up );
		const double		 hr_frac = set_diff_frac( my_hr, hr_coords );
		std::printf( "[hr scaffold]     got %zu vs ref %d, sym-diff %.4f%% -> %s\n", my_hr.size() / 3, Lhr, 100.0 * hr_frac, hr_frac <= 5e-4 ? "OK" : "FAIL" );
		if( hr_frac > 5e-4 )
			++n_fail;

		// ── 1b. the shipped budget path, at both cascade tiers ────────────────
		// The check above quantizes with a local copy of the formula; this one
		// goes through the code the pipeline actually runs, so a change to
		// cascade_tokens.h that moves the 1024 tier fails here.
		{
			std::vector<int32_t>	   sel_1024;
			const t2cascade::selection s1024 = t2cascade::select_scaffold( got_up, 512, 1024, t2cascade::default_max_num_tokens, sel_1024 );
			const bool				   same	 = s1024.resolution == 1024 && sel_1024 == my_hr;
			std::printf( "[budget 1024]     res %d, %d tokens, %d reductions -> %s\n", s1024.resolution, s1024.tokens, s1024.reductions, same ? "OK" : "FAIL" );
			if( !same )
				++n_fail;

			std::vector<int32_t>	   sel_1536;
			const t2cascade::selection s1536 = t2cascade::select_scaffold( got_up, 512, 1536, t2cascade::default_max_num_tokens, sel_1536 );
			std::printf( "[budget 1536]     res %d (grid %d), %d tokens, %d reductions\n", s1536.resolution, s1536.grid, s1536.tokens, s1536.reductions );

			// Invariants that hold with or without a 1536 reference in the dump.
			bool inv = s1536.resolution >= 1024 && s1536.resolution <= 1536 && s1536.resolution % 128 == 0 && s1536.grid == s1536.resolution / 16 &&
					   ( s1536.tokens < t2cascade::default_max_num_tokens || s1536.resolution == 1024 ) && s1536.tokens >= s1024.tokens;
			int mx = -1;
			for( int32_t v : sel_1536 )
				if( v > mx )
					mx = v;
			inv = inv && mx < s1536.grid;
			std::printf( "[budget 1536 inv] max coord %d < %d, fits budget or floored -> %s\n", mx, s1536.grid, inv ? "OK" : "FAIL" );
			if( !inv )
				++n_fail;

			// Tight gate when the dump was regenerated with the 1536 capture.
			std::vector<float> ref_1536f, ref_res_1536;
			if( ref.load( "hr_coords_1536", ref_1536f ) && ref.load( "hr_resolution_1536", ref_res_1536 ) && !ref_res_1536.empty() ) {
				const int			 want_res = ( int )ref_res_1536[0];
				std::vector<int32_t> ref_1536 = coords_xyz( ref_1536f );
				const double		 f		  = set_diff_frac( sel_1536, ref_1536 );
				const bool			 ok		  = s1536.resolution == want_res && f <= 5e-4;
				std::printf( "[hr scaffold 1536] res %d vs ref %d, got %zu vs ref %zu, sym-diff %.4f%% -> %s\n",
					s1536.resolution,
					want_res,
					sel_1536.size() / 3,
					ref_1536.size() / 3,
					100.0 * f,
					ok ? "OK" : "FAIL" );
				if( !ok )
					++n_fail;
			} else {
				std::printf( "[hr scaffold 1536] no reference in the dump (regenerate with scripts/dump_cascade_reference.py), invariants only\n" );
			}
		}
	}

	// ── 2. HR flow forward at t=500 on the reference HR scaffold ──────────────
	std::vector<float> want_hr_fwd;
	const bool		   have_hr_fwd = have_hr_chain && ref.load( "hr_flow_t500_out", want_hr_fwd );
	if( !have_hr_fwd ) {
		std::printf( "\n[hr flow t500]    not in the dump (--skip-hr-sampler), section skipped\n" );
	} else {
		// Force CPU: at 10k+ HR tokens the attention exceeds the exact-path
		// threshold and uses flash, and GPU flash (F16-MMA) is ~1e-2 vs the
		// exact fp32 reference. CPU flash is exact-matching (~3e-4), so it
		// gives a meaningful tight gate. TRELLIS2_CASCADE_GPU overrides.
		const char*				  dev  = std::getenv( "TRELLIS2_CASCADE_GPU" ) ? nullptr : "cpu";
		trellis2_slat_flow_model* flow = trellis2_slat_flow_load( flow1024_path, true, &err, dev );
		if( !flow ) {
			std::fprintf( stderr, "1024 flow load failed: %s\n", err.c_str() );
			trellis2_shape_dec_free( dec );
			return 1;
		}
		std::printf( "1024 flow backend: %s\n", trellis2_slat_flow_backend_name( flow ) );
		std::vector<float> got( ( size_t )Lhr * 32 );
		if( !trellis2_slat_flow_forward( flow, hr_noise.data(), Lhr, hr_coords.data(), 500.0f, cond1024.data(), Lkv1024, 1024, got.data(), &err ) ) {
			std::fprintf( stderr, "HR flow forward failed: %s\n", err.c_str() );
			trellis2_slat_flow_free( flow );
			trellis2_shape_dec_free( dec );
			return 1;
		}
		t2_parity::compare_stats st;
		t2_parity::compare( got, want_hr_fwd, "hr_flow_t500_out", 2e-3, 2e-3, &st );
		if( st.rel_l2 > 3e-3 ) {
			std::printf( "  -> HR forward rel_l2 %.4g > 3e-3, FAIL\n", st.rel_l2 );
			++n_fail;
		}

		// optional: full HR sampler (expensive at ~40k tokens; off by default)
		if( std::getenv( "TRELLIS2_CASCADE_SAMPLE" ) ) {
			trellis2_ss_sampler_params P;
			P.steps					= 12;
			P.guidance_strength		= 7.5f;
			P.guidance_rescale		= 0.5f;
			P.guidance_interval_min = 0.6f;
			P.guidance_interval_max = 1.0f;
			P.rescale_t				= 3.0f;
			std::vector<float> sampled( ( size_t )Lhr * 32 ), wslat;
			if( trellis2_slat_flow_sample( flow, Lhr, hr_coords.data(), cond1024.data(), Lkv1024, 1024, &P, hr_noise.data(), true, sampled.data(), &err ) ) {
				ref.load( "hr_slat", wslat );
				t2_parity::compare_stats ss;
				t2_parity::compare( sampled, wslat, "hr_slat(sampled)", 5e-2, 5e-2, &ss );
			}
		}
		trellis2_slat_flow_free( flow );
	}

	// ── 3. final 1024^3 decode of the reference HR slat ──────────────────────
	// The 1024^3 decode is the SAME decoder validated exactly at the 512 tier
	// (test_slat, levels 0-4 at rel-L2 5e-7) applied to more voxels, and the
	// end-to-end demo exercises it directly. It also transiently needs ~14 GB of
	// host RAM (the finest up-block's conv output is held in both the graph and
	// the readback), so it is gated behind TRELLIS2_CASCADE_DECODE — enable it
	// on a big-RAM box to also gate out7 here.
	if( !std::getenv( "TRELLIS2_CASCADE_DECODE" ) ) {
		trellis2_shape_dec_free( dec );
		std::printf( "\n(1024^3 decode gate skipped; set TRELLIS2_CASCADE_DECODE to enable)\n" );
		std::printf( "total failures: %d\nRESULT: %s\n", n_fail, n_fail ? "FAIL" : "PASS" );
		return n_fail ? 1 : 0;
	}
	std::vector<float> hr_slat;
	if( !ref.load( "hr_slat", hr_slat ) ) {
		std::printf( "\n[1024^3 decode]   hr_slat not in the dump (--skip-hr-sampler), section skipped\n" );
		trellis2_shape_dec_free( dec );
		std::printf( "total failures: %d\nRESULT: %s\n", n_fail, n_fail ? "FAIL" : "PASS" );
		return n_fail ? 1 : 0;
	}
	// Compare only the final 7-channel output (taps=nullptr): the per-level
	// intermediates would hold multiple GB at 1024^3 and the decoder's level
	// logic is already validated exactly at the 512 tier (test_slat, same
	// decoder). out7 is the load-bearing gate.
	std::vector<float>	 out_feats;
	std::vector<int32_t> out_coords;
	if( !trellis2_shape_dec_decode( dec, hr_slat.data(), Lhr, hr_coords.data(), out_feats, out_coords, nullptr, &err ) ) {
		std::fprintf( stderr, "HR decode failed: %s\n", err.c_str() );
		trellis2_shape_dec_free( dec );
		return 1;
	}
	trellis2_shape_dec_free( dec );

	std::printf( "HR decode: %zu output voxels\n", out_coords.size() / 3 );
	std::vector<float> ref_out7;
	ref.load( "out7", ref_out7 );
	const size_t a = out_feats.size(), b = ref_out7.size();
	const size_t big = a > b ? a : b, sml = a > b ? b : a;
	if( a != b ) {
		const double frac = ( double )( big - sml ) / ( double )big;
		if( frac > 5e-4 ) {
			std::printf( "[out7] SIZE MISMATCH %zu vs %zu (%.3f%%) -> FAIL\n", a, b, 100.0 * frac );
			++n_fail;
		} else {
			std::printf( "[out7] near-match (%zu vs %zu, %.4f%% subdivision boundary flip) -> OK\n", a, b, 100.0 * frac );
		}
	} else {
		t2_parity::compare_stats st;
		t2_parity::compare( out_feats, ref_out7, "out7", 2e-3, 2e-3, &st );
		if( st.rel_l2 > 2e-2 ) {
			std::printf( "  -> out7 rel_l2 %.4g > 2e-2, FAIL\n", st.rel_l2 );
			++n_fail;
		}
	}

	std::printf( "\ntotal failures: %d\n", n_fail );
	std::printf( "RESULT: %s\n", n_fail ? "FAIL" : "PASS" );
	return n_fail ? 1 : 0;
}
