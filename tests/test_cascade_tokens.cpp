// The cascade HR token budget (src/cascade_tokens.h) as a pure unit check:
// no models, no GPU, no fixtures — it runs in the default (-LE model) set.
//
// What it pins down:
//   1. the quantization formula itself, including that it TRUNCATES (upstream's
//      .int()) rather than rounds, and that the top source coordinate lands on
//      hr_grid-1 rather than out of bounds;
//   2. the reduction loop: for every budget, the resolution it picks is the
//      first one whose deduplicated scaffold fits — derived independently here
//      from per-grid counts rather than asserted against baked-in numbers;
//   3. the 1024 floor, which is what keeps a 1536 request from ever being
//      worse than a 1024 one;
//   4. that T2_PIPE_1024 still produces exactly the set the pre-1536 port's
//      inline quantization produced (an independent oracle, not the same code).

#include "cascade_tokens.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_set>
#include <vector>

namespace
{

int n_fail = 0;

void check( bool ok, const char* what )
{
	std::printf( "[%s] %s\n", ok ? "OK  " : "FAIL", what );
	if( !ok )
		++n_fail;
}

void check_eq( long long got, long long want, const char* what )
{
	const bool ok = got == want;
	std::printf( "[%s] %s: got %lld, want %lld\n", ok ? "OK  " : "FAIL", what, got, want );
	if( !ok )
		++n_fail;
}

// The quantization exactly as trellis2_capi.cpp spelled it inline before the
// 1536 tier moved it into a header. Deliberately duplicated: an oracle that
// shares the implementation proves nothing.
std::vector<int32_t> legacy_quantize_64( const std::vector<int32_t>& up )
{
	std::unordered_set<uint64_t> seen;
	std::vector<int32_t>		 out;
	auto						 key = []( int32_t a, int32_t b, int32_t c ) { return ( ( uint64_t )( uint32_t )a << 40 ) | ( ( uint64_t )( uint32_t )b << 20 ) | ( uint64_t )( uint32_t )c; };
	for( size_t i = 0; i < up.size(); i += 3 ) {
		int32_t qx = ( int32_t )( ( up[i] + 0.5f ) / 512 * 64 );
		int32_t qy = ( int32_t )( ( up[i + 1] + 0.5f ) / 512 * 64 );
		int32_t qz = ( int32_t )( ( up[i + 2] + 0.5f ) / 512 * 64 );
		if( seen.insert( key( qx, qy, qz ) ).second ) {
			out.push_back( qx );
			out.push_back( qy );
			out.push_back( qz );
		}
	}
	return out;
}

// A deterministic stand-in for the shape decoder's 512^3 candidate cloud: a
// thick spherical shell, which is what an upsampled scaffold looks like — a
// surface, not a volume. Seeded LCG, so every run and platform sees the same
// set (float truncation near a cell boundary must not be able to drift).
std::vector<int32_t> shell_cloud( int n, float r_min, float r_max )
{
	std::vector<int32_t> out;
	out.reserve( ( size_t )n * 3 );
	uint32_t s	  = 0x2b1536u;
	auto	 next = [&]() {
		 s = s * 1664525u + 1013904223u;
		 return ( float )( ( s >> 8 ) & 0xffffff ) / ( float )0x1000000; // [0,1)
	};
	auto	 clamp512 = []( float v ) {
		 const int c = ( int )v;
		 return ( int32_t )( c < 0 ? 0 : ( c > 511 ? 511 : c ) );
	};
	for( int i = 0; i < n; ++i ) {
		// Uniform-ish direction from two randoms, then a radius in the shell.
		const float z	= 2.0f * next() - 1.0f;
		const float phi = 6.28318530718f * next();
		const float rho = 1.0f - z * z > 0.0f ? std::sqrt( 1.0f - z * z ) : 0.0f;
		const float rad = r_min + ( r_max - r_min ) * next();
		const float cx	= 256.0f + rad * rho * std::cos( phi );
		const float cy	= 256.0f + rad * rho * std::sin( phi );
		const float cz	= 256.0f + rad * z;
		out.push_back( clamp512( cx ) );
		out.push_back( clamp512( cy ) );
		out.push_back( clamp512( cz ) );
	}
	return out;
}

int max_coord( const std::vector<int32_t>& c )
{
	int m = -1;
	for( int32_t v : c )
		if( v > m )
			m = v;
	return m;
}

} // namespace

int main()
{
	using namespace t2cascade;

	// ── 1. the formula ───────────────────────────────────────────────────────
	{
		// (c + 0.5) / 512 * 96, truncated. All of these are exact in binary
		// floating point (512 is a power of two), so the expected values are
		// not a tolerance question.
		std::vector<int32_t> in { 0, 5, 511, 8, 7, 256, 511, 511, 0 };
		std::vector<int32_t> got;
		quantize_scaffold( in, 512, 96, got );
		check_eq( ( long long )( got.size() / 3 ), 3, "quantize at 96: three distinct cells" );
		check_eq( got[0], 0, "c=0 -> 0 (0.09375 truncates down)" );
		check_eq( got[1], 1, "c=5 -> 1 (1.03125 truncates down, does not round to 1 by luck)" );
		check_eq( got[2], 95, "c=511 -> 95 (top source cell stays in bounds)" );
		check_eq( got[3], 1, "c=8 -> 1" );
		check_eq( got[4], 1, "c=7 -> 1 (7 and 8 collapse: truncation, not rounding)" );
		check_eq( got[5], 48, "c=256 -> 48" );

		// Rounding would have sent 5 -> 1 as well, so pin a case where the two
		// differ: (c+0.5)/512*96 = 1.6875 at c=8 for grid 96? No — use grid 64,
		// where c=12 gives 1.5625: truncation 1, rounding 2.
		std::vector<int32_t> in2 { 12, 12, 12 };
		std::vector<int32_t> got2;
		quantize_scaffold( in2, 512, 64, got2 );
		check_eq( got2[0], 1, "c=12 at grid 64 -> 1 (truncated; rounding would give 2)" );
	}

	// ── 2. dedup keeps first-seen order and drops repeats ────────────────────
	{
		std::vector<int32_t> in { 3, 3, 3, 4, 4, 4, 300, 300, 300, 3, 3, 3 };
		std::vector<int32_t> got;
		quantize_scaffold( in, 512, 64, got );
		check_eq( ( long long )( got.size() / 3 ), 2, "dedup: 3,3,3 and 4,4,4 share a cell; 300 is its own" );
		check_eq( got[0], 0, "first-seen order preserved (cell of 3)" );
		check_eq( got[3], 37, "second cell is 300 -> 37" );
	}

	// ── 3. the reduction loop against independently derived counts ───────────
	// Big enough that the five grids give clearly different token counts, small
	// enough that the whole test stays well under a second: this runs in the
	// asset-free suite, which people run constantly.
	const std::vector<int32_t> cloud = shell_cloud( 120000, 150.0f, 235.0f );
	std::printf( "\nsynthetic 512^3 shell: %zu candidate coords\n", cloud.size() / 3 );

	const int				   grids[] = { 96, 88, 80, 72, 64 }; // 1536 .. 1024, step 128
	int						   counts[5];
	for( int i = 0; i < 5; ++i ) {
		std::vector<int32_t> q;
		quantize_scaffold( cloud, 512, grids[i], q );
		counts[i] = ( int )( q.size() / 3 );
		std::printf( "  grid %2d (res %4d): %6d tokens\n", grids[i], grids[i] * 16, counts[i] );
		check( max_coord( q ) < grids[i], "quantized coords stay inside the grid" );
	}
	check( counts[0] > counts[4], "coarser grids yield fewer tokens (the loop can actually help)" );

	// For every budget, the answer must be the first resolution in
	// 1536,1408,...,1024 whose count fits — with 1024 taken unconditionally.
	// The budgets tested are the boundaries themselves (count-1, count, count+1
	// per grid) plus the extremes: off-by-one in the `<` is the failure this
	// has to catch, and a coarse sweep would step straight over it.
	{
		std::vector<int> budgets { 1, 2, counts[0] * 2 };
		for( int i = 0; i < 5; ++i ) {
			budgets.push_back( counts[i] - 1 );
			budgets.push_back( counts[i] );
			budgets.push_back( counts[i] + 1 );
		}
		bool all_ok = true;
		for( int budget : budgets ) {
			int want_idx = 4; // the floor, if nothing fits
			for( int i = 0; i < 5; ++i ) {
				if( counts[i] < budget ) {
					want_idx = i;
					break;
				}
			}
			std::vector<int32_t> out;
			const selection		 s = select_scaffold( cloud, 512, 1536, budget, out );
			if( s.resolution != grids[want_idx] * 16 || s.grid != grids[want_idx] || s.tokens != counts[want_idx] || s.reductions != want_idx || ( int )( out.size() / 3 ) != counts[want_idx] ) {
				std::printf( "  budget %d: got res %d/grid %d/tokens %d/reductions %d, want res %d/tokens %d\n",
					budget,
					s.resolution,
					s.grid,
					s.tokens,
					s.reductions,
					grids[want_idx] * 16,
					counts[want_idx] );
				all_ok = false;
			}
		}
		check( all_ok, "1536 request lands on the first resolution that fits, at every budget boundary" );
	}

	// ── 4. the 1024 floor ────────────────────────────────────────────────────
	{
		std::vector<int32_t> out;
		const selection		 s = select_scaffold( cloud, 512, 1536, 1, out );
		check_eq( s.resolution, 1024, "an unsatisfiable budget stops at the 1024 floor, it does not loop past it" );
		check_eq( s.grid, 64, "floor scaffold is 64^3" );
		check_eq( s.reductions, 4, "1536 -> 1024 is four steps of 128" );

		// A 1024 request never reduces, whatever the budget — that is the
		// property that makes 1536 safe to offer: it can degrade to 1024, and
		// 1024 itself is never degraded.
		std::vector<int32_t> out1024;
		const selection		 s1 = select_scaffold( cloud, 512, 1024, 1, out1024 );
		check_eq( s1.resolution, 1024, "1024 request with an impossible budget still returns 1024" );
		check_eq( s1.reductions, 0, "1024 request never reduces" );

		// And a request below the floor terminates instead of running away.
		std::vector<int32_t> out512;
		const selection		 s2 = select_scaffold( cloud, 512, 512, 1, out512 );
		check_eq( s2.resolution, 512, "a sub-floor request terminates at itself" );
	}

	// ── 5. T2_PIPE_1024 is byte-identical to the pre-1536 inline code ────────
	{
		std::vector<int32_t> out;
		const selection		 s	   = select_scaffold( cloud, 512, 1024, default_max_num_tokens, out );
		std::vector<int32_t> legacy = legacy_quantize_64( cloud );
		check_eq( s.resolution, 1024, "default budget at 1024 stays at 1024" );
		check( out == legacy, "1024 scaffold matches the legacy inline quantization coord for coord, in order" );
		std::printf( "  1024 scaffold: %zu voxels (legacy %zu)\n", out.size() / 3, legacy.size() / 3 );
	}

	std::printf( "\ntotal failures: %d\nRESULT: %s\n", n_fail, n_fail ? "FAIL" : "PASS" );
	return n_fail ? 1 : 0;
}
