// test_ss_flow_forward — validate the C++ SS-flow DiT forward pass against the
// PyTorch float32 reference produced by ref_ss_flow.py.
//
//   usage: test_ss_flow_forward <ss_flow_dit_f32.gguf> <ss_flow_ref.bin> [rel_tol]
//
// Reads x/t/cond and the reference output from ss_flow_ref.bin, runs the C++
// forward with the f32 weights, and reports max abs / relative error. Exits
// nonzero if the relative L2 error exceeds the tolerance (default 2e-3).

#include "trellis2.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool read_exact( std::ifstream& f, void* p, size_t n )
{
	return ( bool )f.read( reinterpret_cast<char*>( p ), ( std::streamsize )n );
}

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <f32.gguf> <ss_flow_ref.bin> [rel_tol]\n", argv[0] );
		return 2;
	}
	const std::string gguf_path = argv[1];
	const std::string ref_path	= argv[2];

	{
		std::ifstream _a( gguf_path ), _b( ref_path );
		if( !_a.good() || !_b.good() ) {
			std::fprintf( stderr, "missing input file(s), skipping\n" );
			return 77;
		}
	}
	const double  rel_tol = ( argc > 3 ) ? std::atof( argv[3] ) : 2e-3;

	// ── read the reference bundle ────────────────────────────────────────────
	std::ifstream f( ref_path, std::ios::binary );
	char		  magic[8];
	if( !f || !read_exact( f, magic, 8 ) || std::memcmp( magic, "SSFREF01", 8 ) != 0 ) {
		std::fprintf( stderr, "error: bad/missing ref file %s\n", ref_path.c_str() );
		return 1;
	}
	int32_t dims[5];
	float	t = 0.0f;
	read_exact( f, dims, sizeof( dims ) );
	read_exact( f, &t, sizeof( t ) );
	const int		   R = dims[0], Cin = dims[1], Cout = dims[2], Lkv = dims[3], Cctx = dims[4];
	const size_t	   N = ( size_t )R * R * R;

	std::vector<float> x( ( size_t )Cin * N );
	std::vector<float> cond( ( size_t )Lkv * Cctx );
	std::vector<float> ref( ( size_t )Cout * N );
	read_exact( f, x.data(), x.size() * sizeof( float ) );
	read_exact( f, cond.data(), cond.size() * sizeof( float ) );
	if( !read_exact( f, ref.data(), ref.size() * sizeof( float ) ) ) {
		std::fprintf( stderr, "error: ref file truncated\n" );
		return 1;
	}
	std::printf( "ref: R=%d Cin=%d Cout=%d Lkv=%d Cctx=%d t=%.3f\n", R, Cin, Cout, Lkv, Cctx, t );

	// ── load weights + run the C++ forward ───────────────────────────────────
	std::string				err;
	trellis2_ss_flow_model* m = trellis2_ss_flow_load( gguf_path, /*load_tensors*/ true, &err );
	if( !m ) {
		std::fprintf( stderr, "load error: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "backend: %s\n", trellis2_ss_flow_backend_name( m ) );

	std::vector<float> out( ( size_t )Cout * N, 0.0f );
	if( !trellis2_ss_flow_forward( m, x.data(), t, cond.data(), Lkv, Cctx, out.data(), &err ) ) {
		std::fprintf( stderr, "forward error: %s\n", err.c_str() );
		trellis2_ss_flow_free( m );
		return 1;
	}
	trellis2_ss_flow_free( m );

	// ── compare ──────────────────────────────────────────────────────────────
	double max_abs = 0.0, sse = 0.0, ref_sq = 0.0;
	int	   max_i = 0;
	for( size_t i = 0; i < out.size(); ++i ) {
		const double d = ( double )out[i] - ( double )ref[i];
		if( std::fabs( d ) > max_abs ) {
			max_abs = std::fabs( d );
			max_i	= ( int )i;
		}
		sse += d * d;
		ref_sq += ( double )ref[i] * ( double )ref[i];
	}
	const double rel_l2 = std::sqrt( sse ) / ( std::sqrt( ref_sq ) + 1e-30 );

	std::printf( "out: min/max checked over %zu elems\n", out.size() );
	std::printf( "max abs err   : %.3e  (at %d: cpp=%.6f ref=%.6f)\n", max_abs, max_i, out[max_i], ref[max_i] );
	std::printf( "rel L2 err    : %.3e  (tol %.1e)\n", rel_l2, rel_tol );

	if( rel_l2 > rel_tol ) {
		std::printf( "RESULT: FAIL\n" );
		return 1;
	}
	std::printf( "RESULT: PASS\n" );
	return 0;
}
