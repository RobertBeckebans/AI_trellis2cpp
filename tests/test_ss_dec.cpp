// test_ss_dec — validate the C++ SS decoder against the PyTorch reference
// produced by ref_ss_dec.py.
//
//   usage: test_ss_dec <ss_dec_f32.gguf> <ss_dec_ref.bin> [rel_tol]
//
// Reads the input latent z_s and the reference occupancy logits, runs the C++
// decoder on the identical latent, and reports max abs / relative L2 error. Also
// checks sign agreement at 0 — the occupancy scaffold is logit>0, so the sign
// map is what ultimately matters. Default tolerance 3e-2.

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
		std::fprintf( stderr, "usage: %s <ss_dec_f32.gguf> <ss_dec_ref.bin> [rel_tol]\n", argv[0] );
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
	if( !f || !rd( f, magic, 8 ) || std::memcmp( magic, "SSDEC001", 8 ) != 0 ) {
		std::fprintf( stderr, "error: bad/missing ref file %s\n", ref_path.c_str() );
		return 1;
	}
	int32_t hdr[4];
	rd( f, hdr, sizeof( hdr ) );
	const int		   Cin = hdr[0], Rin = hdr[1], Oc = hdr[2], Rout = hdr[3];
	const size_t	   n_in	 = ( size_t )Cin * Rin * Rin * Rin;
	const size_t	   n_out = ( size_t )Oc * Rout * Rout * Rout;

	std::vector<float> latent( n_in ), ref( n_out );
	rd( f, latent.data(), latent.size() * sizeof( float ) );
	if( !rd( f, ref.data(), ref.size() * sizeof( float ) ) ) {
		std::fprintf( stderr, "error: ref truncated\n" );
		return 1;
	}
	std::printf( "ref: latent[%d,%d^3] -> logits[%d,%d^3]\n", Cin, Rin, Oc, Rout );

	std::string			   err;
	trellis2_ss_dec_model* m = trellis2_ss_dec_load( gguf_path, true, &err );
	if( !m ) {
		std::fprintf( stderr, "load error: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "backend: %s\n", trellis2_ss_dec_backend_name( m ) );

	const trellis2_ss_dec_hparams hp = trellis2_ss_dec_hparams_of( m );
	if( hp.latent_channels != Cin || hp.res_in() != Rin || hp.out_channels != Oc || hp.res_out() != Rout ) {
		std::fprintf( stderr, "error: model/ref shape mismatch\n" );
		trellis2_ss_dec_free( m );
		return 1;
	}

	std::vector<float> out( n_out, 0.0f );
	if( !trellis2_ss_dec_decode( m, latent.data(), out.data(), &err ) ) {
		std::fprintf( stderr, "decode error: %s\n", err.c_str() );
		trellis2_ss_dec_free( m );
		return 1;
	}
	trellis2_ss_dec_free( m );

	double max_abs = 0.0, sse = 0.0, ref_sq = 0.0;
	size_t sign_agree = 0;
	for( size_t i = 0; i < n_out; ++i ) {
		const double d = ( double )out[i] - ( double )ref[i];
		max_abs		   = std::fmax( max_abs, std::fabs( d ) );
		sse += d * d;
		ref_sq += ( double )ref[i] * ( double )ref[i];
		if( ( out[i] > 0.0f ) == ( ref[i] > 0.0f ) )
			++sign_agree;
	}
	const double rel_l2	  = std::sqrt( sse ) / ( std::sqrt( ref_sq ) + 1e-30 );
	const double sign_pct = 100.0 * ( double )sign_agree / ( double )n_out;

	std::printf( "max abs err : %.3e\n", max_abs );
	std::printf( "rel L2 err  : %.3e  (tol %.1e)\n", rel_l2, rel_tol );
	std::printf( "sign agree  : %.3f%%  (occupancy is logit>0)\n", sign_pct );

	if( rel_l2 > rel_tol ) {
		std::printf( "RESULT: FAIL\n" );
		return 1;
	}
	std::printf( "RESULT: PASS\n" );
	return 0;
}
