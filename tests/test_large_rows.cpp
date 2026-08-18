// Locate the row count at which a GPU backend stops agreeing with the CPU.
//
// The 1024^3 cascade decode produces garbage past output row 2^21 = 2097152 on
// ROCm/HIP (the mesh loses every face beyond that slab; see docs/bugs/). The
// finest decoder level runs its submanifold conv over an [64, L] matrix with
// L ~ 2.9M rows, so 2^21 rows x 1024 = 2^31 elements is exactly where a signed
// 32-bit element index would wrap.
//
// This probe runs each op of that level in isolation on the GPU and on the CPU
// with identical inputs, and reports the FIRST row where they diverge. If the
// hypothesis holds, exactly one op reports first_bad_row = 2097152.
//
// The ceilings are not going to be fixed here, so failing merely because the
// backend is broken would leave a permanently red test. What this guards instead
// is the assumption the workaround rests on: trellis2.cpp's mul_mat_max_rows()
// hard-codes these numbers to decide where the decoder has to split. A ceiling
// that moved would silently invalidate that, so a *different* break row fails
// while the known one passes. A backend that stopped breaking also passes — the
// split just becomes unnecessarily conservative.
//
// --sweep additionally varies the weight width, because mul_mat_max_rows()
// states a row count per weight TYPE and the measurement behind it was taken at
// one width. Measured 2026-08-18 on gfx1201 the ceiling rises with the width
// (f32: 2^19 at C<=64, 2^20 at C>=128), so the stated numbers are the minimum
// and the split is conservative at the encoder's wider convs rather than short.
// Only a width that breaks EARLIER than the stated number fails.
//
// Usage: test_large_rows [L] [--sweep]   (default L 2200000, just above 2^21)
// Exit:  0 as expected, 1 an assumption broke, 77 no GPU backend.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{

// Channels of the finest shape-decoder level (hp.channels[4]). The pass/fail
// gate runs at this width, because it is the shape mul_mat_max_rows() was
// measured at and therefore the only one it can honestly claim.
//
// --sweep varies it, and that is the more interesting run. trellis2.cpp states
// the ceiling as a ROW COUNT for every width, while the powers of two it lands
// on suggest a grid dimension times a tile height. If the real bound is on
// elements or bytes rather than rows, the row count has to fall as the row gets
// wider — and a limit measured at 64 channels is then wrong for the encoder's
// 128- and 256-channel convs in the unsafe direction.
const int C_DEFAULT = 64;

enum op_id { OP_GET_ROWS, OP_MUL_MASK, OP_MUL_MAT, OP_MUL_MAT_F16, OP_MUL_MAT_BF16, OP_ADD, OP_NORM, OP_SILU, OP_REPEAT, OP_COUNT };

const char* op_name( int op )
{
	switch( op ) {
		case OP_GET_ROWS:
			return "get_rows";
		case OP_MUL_MASK:
			return "mul(mask bcast)";
		case OP_MUL_MAT:
			return "mul_mat f32";
		case OP_MUL_MAT_F16:
			return "mul_mat f16"; // what the shipped GGUF actually uses
		case OP_MUL_MAT_BF16:
			return "mul_mat bf16"; // decides what the fix may assume for bf16
		case OP_ADD:
			return "add";
		case OP_NORM:
			return "norm";
		case OP_SILU:
			return "silu";
		case OP_REPEAT:
			return "repeat";
	}
	return "?";
}

// The row where this op is known to start dropping output, mirroring
// mul_mat_max_rows() in trellis2.cpp. 0 means "must never diverge".
int64_t expected_break( int op )
{
	switch( op ) {
		case OP_MUL_MAT:
			return ( int64_t )1 << 19;
		case OP_MUL_MAT_F16:
		case OP_MUL_MAT_BF16:
			return ( int64_t )1 << 21;
		default:
			return 0;
	}
}

// Data movement must match bit-exactly; arithmetic may legitimately differ in
// accumulation order / FMA contraction, so only gross divergence counts.
double op_tol( int op )
{
	switch( op ) {
		case OP_MUL_MAT:
		case OP_MUL_MAT_F16:
			return 1e-2; // f16 weights carry ~1e-3 of their own
		case OP_MUL_MAT_BF16:
			return 1e-1; // bf16 has 8 mantissa bits, so much coarser
		case OP_NORM:
		case OP_SILU:
			return 1e-4;
		default:
			return 0.0; // get_rows / mul / add / repeat
	}
}

// Deterministic inputs, so both backends see bit-identical data.
float lcg( uint32_t& s )
{
	s = s * 1664525u + 1013904223u;
	return ( float )( ( int32_t )( s >> 8 ) % 2000 - 1000 ) * 0.001f;
}

// Build + run one op on `backend`, returning the flat result.
// Every graph mirrors how trellis2.cpp's shape_dec_run drives that op.
bool run_op( ggml_backend_t		backend,
	int							op,
	int							C,
	int64_t						L,
	const std::vector<float>&	xdata,
	const std::vector<int32_t>& idata,
	const std::vector<float>&	mdata,
	const std::vector<float>&	wdata,
	std::vector<float>&			out )
{
	const size_t	 gsize = 256;
	ggml_init_params ip { ggml_tensor_overhead() * gsize + ggml_graph_overhead_custom( gsize, false ), nullptr, true };
	ggml_context*	 ctx = ggml_init( ip );
	if( !ctx )
		return false;
	ggml_cgraph* gf = ggml_new_graph_custom( ctx, gsize, false );

	ggml_tensor *x = nullptr, *idx = nullptr, *mask = nullptr, *w = nullptr, *y = nullptr;

	if( op == OP_REPEAT ) {
		// The up-block skip: [1, C/4, L] repeat_interleave'd to [4, C/4, L].
		x = ggml_new_tensor_3d( ctx, GGML_TYPE_F32, 1, C / 4, L );
		ggml_set_input( x );
		y = ggml_repeat( ctx, x, ggml_new_tensor_3d( ctx, GGML_TYPE_F32, 4, C / 4, L ) );
	} else {
		x = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, C, L );
		ggml_set_input( x );
		switch( op ) {
			case OP_GET_ROWS:
				idx = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, L );
				ggml_set_input( idx );
				y = ggml_get_rows( ctx, x, idx );
				break;
			case OP_MUL_MASK:
				mask = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, L );
				ggml_set_input( mask );
				y = ggml_mul( ctx, x, mask );
				break;
			case OP_MUL_MAT:
				w = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, C, C );
				ggml_set_input( w );
				y = ggml_mul_mat( ctx, w, x );
				break;
			case OP_MUL_MAT_F16:
				w = ggml_new_tensor_2d( ctx, GGML_TYPE_F16, C, C );
				ggml_set_input( w );
				y = ggml_mul_mat( ctx, w, x );
				break;
			case OP_MUL_MAT_BF16:
				w = ggml_new_tensor_2d( ctx, GGML_TYPE_BF16, C, C );
				ggml_set_input( w );
				y = ggml_mul_mat( ctx, w, x );
				break;
			case OP_ADD:
				y = ggml_add( ctx, x, x );
				break;
			case OP_NORM:
				y = ggml_norm( ctx, x, 1e-5f );
				break;
			case OP_SILU:
				y = ggml_silu( ctx, x );
				break;
		}
	}
	y = ggml_cont( ctx, y );
	ggml_set_output( y );
	ggml_build_forward_expand( gf, y );

	ggml_gallocr_t alloc = ggml_gallocr_new( ggml_backend_get_default_buffer_type( backend ) );
	if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
		std::printf( "  %-16s alloc failed (needs more memory than the backend allows)\n", op_name( op ) );
		ggml_gallocr_free( alloc );
		ggml_free( ctx );
		return false;
	}

	ggml_backend_tensor_set( x, xdata.data(), 0, ggml_nbytes( x ) );
	if( idx )
		ggml_backend_tensor_set( idx, idata.data(), 0, ggml_nbytes( idx ) );
	if( mask )
		ggml_backend_tensor_set( mask, mdata.data(), 0, ggml_nbytes( mask ) );
	if( w && w->type == GGML_TYPE_F16 ) {
		std::vector<ggml_fp16_t> wf16( ( size_t )C * C );
		ggml_fp32_to_fp16_row( wdata.data(), wf16.data(), ( int64_t )C * C );
		ggml_backend_tensor_set( w, wf16.data(), 0, ggml_nbytes( w ) );
	} else if( w && w->type == GGML_TYPE_BF16 ) {
		std::vector<ggml_bf16_t> wbf16( ( size_t )C * C );
		ggml_fp32_to_bf16_row( wdata.data(), wbf16.data(), ( int64_t )C * C );
		ggml_backend_tensor_set( w, wbf16.data(), 0, ggml_nbytes( w ) );
	} else if( w ) {
		ggml_backend_tensor_set( w, wdata.data(), 0, ggml_nbytes( w ) );
	}

	const bool ok = ggml_backend_graph_compute( backend, gf ) == GGML_STATUS_SUCCESS;
	if( ok ) {
		out.resize( ( size_t )ggml_nelements( y ) );
		ggml_backend_tensor_get( y, out.data(), 0, out.size() * sizeof( float ) );
	}
	ggml_gallocr_free( alloc );
	ggml_free( ctx );
	return ok;
}

// One op at one width: the first row where the GPU stops agreeing with the CPU.
// -1 means no divergence, which is the healthy answer; -2 means it would not run
// at all, which at these sizes is usually the allocator refusing.
int64_t break_row( ggml_backend_t gpu, ggml_backend_t cpu, int op, int C, int64_t L )
{
	uint32_t		   s = 12345;
	std::vector<float> xdata( ( size_t )C * L );
	for( auto& v : xdata )
		v = lcg( s );
	std::vector<int32_t> idata( ( size_t )L );
	for( int64_t i = 0; i < L; ++i )
		idata[( size_t )i] = ( int32_t )( ( i * 7919 + 13 ) % L );
	std::vector<float> mdata( ( size_t )L );
	for( int64_t i = 0; i < L; ++i )
		mdata[( size_t )i] = ( i % 5 ) ? 1.0f : 0.0f;
	std::vector<float> wdata( ( size_t )C * C );
	for( auto& v : wdata )
		v = lcg( s );

	std::vector<float> g, c;
	if( !run_op( gpu, op, C, L, xdata, idata, mdata, wdata, g ) )
		return -2;
	if( !run_op( cpu, op, C, L, xdata, idata, mdata, wdata, c ) )
		return -2;
	if( g.size() != c.size() )
		return -3;

	const double tol = op_tol( op );
	for( size_t i = 0; i < g.size(); ++i ) {
		const double d = ( double )g[i] - ( double )c[i];
		if( ( d < 0 ? -d : d ) > tol )
			return ( int64_t )( i / ( size_t )C );
	}
	return -1;
}

// The ceiling as a function of row width, for the three ops that have one.
// Prints the break as rows, as elements and as bytes, because which of the three
// stays constant across widths IS the question: a constant row count means the
// bound really is on rows and mul_mat_max_rows() is shaped right; a constant
// element or byte count means its per-type row numbers only hold at the width
// they were measured at, and the encoder runs wider than that.
int sweep_widths( ggml_backend_t gpu, ggml_backend_t cpu, int64_t L )
{
	const int widths[] = { 32, 64, 128, 256 };
	const int ops[]	   = { OP_MUL_MAT, OP_MUL_MAT_F16, OP_MUL_MAT_BF16 };
	const int esize[]  = { 4, 2, 2 }; // src0 element size, for the byte column

	std::printf( "\nSweep over row width. L = %" PRId64 ", weight [C, C].\n", L );
	std::printf( "  %-16s %5s  %13s  %14s  %14s\n", "op", "C", "first bad row", "rows*C", "rows*C*size" );

	// Only a ceiling BELOW what trellis2.cpp assumes is a defect. One above it
	// means the split is conservative at that width, which this file already
	// treats as a pass for the single-width case ("a backend that stopped
	// breaking also passes"), and the same reasoning has to apply here or the
	// sweep is permanently red for a safe condition.
	int unsafe = 0;
	for( size_t o = 0; o < sizeof( ops ) / sizeof( ops[0] ); ++o ) {
		const int64_t assumed = expected_break( ops[o] );
		int64_t		  seen	  = -1;
		bool		  varies  = false;
		for( size_t i = 0; i < sizeof( widths ) / sizeof( widths[0] ); ++i ) {
			const int	  w = widths[i];
			const int64_t r = break_row( gpu, cpu, ops[o], w, L );
			if( r == -2 ) {
				std::printf( "  %-16s %5d  %13s\n", op_name( ops[o] ), w, "not run" );
				continue;
			}
			if( r < 0 ) {
				std::printf( "  %-16s %5d  %13s   (none below L)\n", op_name( ops[o] ), w, "no break" );
				continue;
			}
			const char* verdict = r < assumed ? "  <-- BELOW mul_mat_max_rows" : "";
			std::printf( "  %-16s %5d  %13" PRId64 "  %14" PRId64 "  %14" PRId64 "%s\n", op_name( ops[o] ), w, r, r * w, r * w * esize[o], verdict );
			if( r < assumed )
				++unsafe;
			if( seen < 0 )
				seen = r;
			else if( seen != r )
				varies = true;
		}
		if( varies )
			std::printf( "  %-16s  ^ ceiling varies with width; mul_mat_max_rows() states one number\n"
						 "  %-16s    per weight type, so it has to be the LOWEST of these to be safe.\n",
				"",
				"" );
	}

	std::printf( "\n  Note: \"no break\" means none below L = %" PRId64 ", not that there is none.\n", L );
	if( unsafe )
		std::printf( "  %d width(s) break EARLIER than mul_mat_max_rows() assumes -- the split is\n"
					 "  not conservative there, and trellis2.cpp needs the lower number.\n",
			unsafe );
	else
		std::printf( "  No width breaks earlier than mul_mat_max_rows() assumes, so the split is\n"
					 "  conservative everywhere measured.\n" );
	return unsafe;
}

} // namespace

int main( int argc, char** argv )
{
	bool	sweep = false;
	int64_t L	  = 2200000;
	for( int i = 1; i < argc; ++i ) {
		if( std::strcmp( argv[i], "--sweep" ) == 0 )
			sweep = true;
		else
			L = std::atoll( argv[i] );
	}
	if( L <= 0 ) {
		std::fprintf( stderr, "bad L\n" );
		return 1;
	}

	ggml_backend_t gpu = nullptr;
	std::string	   gpu_name;
	for( size_t i = 0; i < ggml_backend_dev_count(); ++i ) {
		ggml_backend_dev_t dev = ggml_backend_dev_get( i );
		if( ggml_backend_dev_type( dev ) == GGML_BACKEND_DEVICE_TYPE_GPU ) {
			gpu = ggml_backend_dev_init( dev, nullptr );
			if( gpu ) {
				const char* d = ggml_backend_dev_description( dev );
				gpu_name	  = d ? d : ggml_backend_dev_name( dev );
				break;
			}
		}
	}
	if( !gpu ) {
		std::printf( "no GPU backend available -- skipping\n" );
		return 77;
	}
	ggml_backend_t cpu = ggml_backend_cpu_init();

	// The gate runs at one width; --sweep varies it afterwards.
	const int	   C = C_DEFAULT;

	std::printf( "GPU: %s\n", gpu_name.c_str() );
	std::printf( "L = %" PRId64 "  (2^21 = 2097152, %s)  matrix [%d, %" PRId64 "] = %" PRId64 " elements\n\n", L, L > 2097152 ? "above" : "below", C, L, ( int64_t )C * L );

	// Shared deterministic inputs.
	uint32_t		   s = 12345;
	std::vector<float> xdata( ( size_t )C * L );
	for( auto& v : xdata )
		v = lcg( s );
	std::vector<int32_t> idata( ( size_t )L );
	for( int64_t i = 0; i < L; ++i )
		idata[( size_t )i] = ( int32_t )( ( i * 7919 + 13 ) % L ); // scattered
	std::vector<float> mdata( ( size_t )L );
	for( int64_t i = 0; i < L; ++i )
		mdata[( size_t )i] = ( i % 5 ) ? 1.0f : 0.0f; // like the neighbor mask
	std::vector<float> wdata( ( size_t )C * C );
	for( auto& v : wdata )
		v = lcg( s );

	int failures = 0, skipped = 0;
	for( int op = 0; op < OP_COUNT; ++op ) {
		std::vector<float> g, c;
		if( !run_op( gpu, op, C, L, xdata, idata, mdata, wdata, g ) ) {
			++skipped;
			continue;
		}
		if( !run_op( cpu, op, C, L, xdata, idata, mdata, wdata, c ) ) {
			++skipped;
			continue;
		}
		if( g.size() != c.size() ) {
			std::printf( "  %-16s SIZE MISMATCH %zu vs %zu\n", op_name( op ), g.size(), c.size() );
			++failures;
			continue;
		}
		// Row width of the result (repeat collapses to 4*C/4 == C per row anyway).
		const int64_t row		= C;
		const double  tol		= op_tol( op );
		int64_t		  first_bad = -1, nbad = 0;
		double		  worst = 0.0;
		for( size_t i = 0; i < g.size(); ++i ) {
			const double d = ( double )g[i] - ( double )c[i];
			const double a = d < 0 ? -d : d;
			if( a > worst )
				worst = a;
			if( a > tol ) {
				++nbad;
				if( first_bad < 0 )
					first_bad = ( int64_t )i;
			}
		}
		const int64_t bad_row  = first_bad < 0 ? -1 : first_bad / row;
		const int64_t expected = expected_break( op );

		if( first_bad < 0 ) {
			if( expected && L > expected ) {
				std::printf( "  %-16s NO BREAK  expected one at row %" PRId64 " -- backend appears "
							 "fixed, the decoder's split is now conservative\n",
					op_name( op ),
					expected );
			} else {
				std::printf( "  %-16s OK        (max|d| = %.3g, tol %g)\n", op_name( op ), worst, tol );
			}
			continue;
		}

		// Powers of two are the interesting ones: they point at a 32-bit index
		// wrapping rather than at accumulated numerical drift.
		char note[64] = "";
		if( ( bad_row & ( bad_row - 1 ) ) == 0 && bad_row > 0 ) {
			int e = 0;
			for( int64_t v = bad_row; v > 1; v >>= 1 )
				++e;
			std::snprintf( note, sizeof( note ), " == 2^%d", e );
		}
		const bool as_expected = bad_row == expected;
		std::printf( "  %-16s %s  first bad row %" PRId64 "%s  (%" PRId64 "/%zu bad, max|d| = %.3g)\n", op_name( op ), as_expected ? "KNOWN   " : "CHANGED!", bad_row, note, nbad, g.size(), worst );
		if( !as_expected ) {
			std::printf( "  %-16s   expected %" PRId64 " (mul_mat_max_rows in trellis2.cpp assumes "
						 "this) -- at elem %" PRId64 ": gpu=%.6g cpu=%.6g\n",
				"",
				expected,
				first_bad,
				( double )g[( size_t )first_bad],
				( double )c[( size_t )first_bad] );
			++failures;
		}
	}

	std::printf( "\n%d op(s) diverged, %d skipped\n", failures, skipped );
	// Opt-in: three more runs per width, and it answers a different question
	// than the gate. Not "did the ceiling move since last time" but "is a row
	// count the right unit for it at all". A ceiling that varies with the width
	// fails on the same principle as a changed break row — trellis2.cpp would be
	// carrying an assumption that does not hold.
	if( sweep )
		failures += sweep_widths( gpu, cpu, L );

	ggml_backend_free( gpu );
	ggml_backend_free( cpu );
	return failures ? 1 : 0;
}
