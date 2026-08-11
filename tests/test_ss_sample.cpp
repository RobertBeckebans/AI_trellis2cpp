// test_ss_sample — validate the C++ flow-Euler sampler against the PyTorch
// reference produced by ref_ss_sample.py.
//
//   usage: test_ss_sample <ss_flow_dit_f32.gguf> <ss_sample_ref.bin> [rel_tol]
//
// Reads noise/cond/params and the reference latent, runs the C++ sampler with
// the same noise and settings, and reports max abs / relative L2 error. Also
// checks sign agreement (the decoder thresholds at 0, so the sign map is what
// ultimately matters). Default tolerance 3e-2.

#include "trellis2.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool rd( std::ifstream& f, void* p, size_t n )
{
	return ( bool )f.read( reinterpret_cast<char*>( p ), ( std::streamsize )n );
}

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <f32.gguf> <ss_sample_ref.bin> [rel_tol]\n", argv[0] );
		return 2;
	}
	const std::string gguf_path = argv[1], ref_path = argv[2];

	{
		std::ifstream _a( gguf_path ), _b( ref_path );
		if( !_a.good() || !_b.good() ) {
			std::fprintf( stderr, "missing input file(s), skipping\n" );
			return 77;
		}
	}
	const double  rel_tol = ( argc > 3 ) ? std::atof( argv[3] ) : 3e-2;

	std::ifstream f( ref_path, std::ios::binary );
	char		  magic[8];
	if( !f || !rd( f, magic, 8 ) || std::memcmp( magic, "SSSAMP01", 8 ) != 0 ) {
		std::fprintf( stderr, "error: bad/missing ref file %s\n", ref_path.c_str() );
		return 1;
	}
	int32_t hdr[5];
	float	pf[6];
	rd( f, hdr, sizeof( hdr ) );
	rd( f, pf, sizeof( pf ) );
	const int		   R = hdr[0], Cin = hdr[1], Lkv = hdr[2], Cctx = hdr[3], steps = hdr[4];
	const size_t	   N = ( size_t )R * R * R;
	const size_t	   n = ( size_t )Cin * N;

	std::vector<float> noise( n ), cond( ( size_t )Lkv * Cctx ), ref( n );
	rd( f, noise.data(), noise.size() * sizeof( float ) );
	rd( f, cond.data(), cond.size() * sizeof( float ) );
	if( !rd( f, ref.data(), ref.size() * sizeof( float ) ) ) {
		std::fprintf( stderr, "error: ref truncated\n" );
		return 1;
	}

	trellis2_ss_sampler_params P;
	P.steps					= steps;
	P.guidance_strength		= pf[0];
	P.guidance_rescale		= pf[1];
	P.guidance_interval_min = pf[2];
	P.guidance_interval_max = pf[3];
	P.rescale_t				= pf[4];
	P.sigma_min				= pf[5];
	P.verbose				= true;
	std::printf( "ref: R=%d Cin=%d Lkv=%d steps=%d gs=%.2f rescale=%.2f interval=[%.2f,%.2f] rescale_t=%.1f\n",
		R,
		Cin,
		Lkv,
		steps,
		P.guidance_strength,
		P.guidance_rescale,
		P.guidance_interval_min,
		P.guidance_interval_max,
		P.rescale_t );

	std::string				err;
	trellis2_ss_flow_model* m = trellis2_ss_flow_load( gguf_path, true, &err );
	if( !m ) {
		std::fprintf( stderr, "load error: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "backend: %s\n", trellis2_ss_flow_backend_name( m ) );

	std::vector<float> out( n, 0.0f );
	if( !trellis2_ss_flow_sample( m, cond.data(), Lkv, Cctx, &P, noise.data(), out.data(), &err ) ) {
		std::fprintf( stderr, "sample error: %s\n", err.c_str() );
		trellis2_ss_flow_free( m );
		return 1;
	}
	trellis2_ss_flow_free( m );

	double max_abs = 0.0, sse = 0.0, ref_sq = 0.0;
	size_t sign_agree = 0;
	for( size_t i = 0; i < n; ++i ) {
		const double d = ( double )out[i] - ( double )ref[i];
		max_abs		   = std::fmax( max_abs, std::fabs( d ) );
		sse += d * d;
		ref_sq += ( double )ref[i] * ( double )ref[i];
		if( ( out[i] > 0.0f ) == ( ref[i] > 0.0f ) )
			++sign_agree;
	}
	const double rel_l2	  = std::sqrt( sse ) / ( std::sqrt( ref_sq ) + 1e-30 );
	const double sign_pct = 100.0 * ( double )sign_agree / ( double )n;

	std::printf( "max abs err : %.3e\n", max_abs );
	std::printf( "rel L2 err  : %.3e  (tol %.1e)\n", rel_l2, rel_tol );
	std::printf( "sign agree  : %.3f%%  (decoder thresholds z_s at 0)\n", sign_pct );

	if( rel_l2 > rel_tol ) {
		std::printf( "RESULT: FAIL\n" );
		return 1;
	}
	std::printf( "RESULT: PASS\n" );
	return 0;
}
