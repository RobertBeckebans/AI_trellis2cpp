// Layer-by-layer validation of the DINOv3 encoder against the PyTorch
// reference dump (scripts/dump_dino_reference.py).
//
// usage: test_dino <dino_f32.gguf> <reference_dino.gguf> [atol] [rtol]
// exits 77 (ctest SKIP) when either input file is missing.

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
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <dino_f32.gguf> <reference_dino.gguf> [atol] [rtol]\n", argv[0] );
		return 2;
	}
	const std::string gguf_path = argv[1];
	const std::string ref_path	= argv[2];
	const double	  atol		= argc > 3 ? std::atof( argv[3] ) : 2e-3;
	const double	  rtol		= argc > 4 ? std::atof( argv[4] ) : 2e-3;

	if( !file_exists( gguf_path ) || !file_exists( ref_path ) ) {
		std::fprintf( stderr, "missing input file(s), skipping\n" );
		return 77;
	}

	t2_parity::baseline ref;
	if( !ref.open( ref_path ) ) {
		std::fprintf( stderr, "failed to open reference gguf %s\n", ref_path.c_str() );
		return 1;
	}

	std::vector<float> pixels;
	if( !ref.load( "pixel_values", pixels ) ) {
		std::fprintf( stderr, "reference has no pixel_values tensor\n" );
		return 1;
	}
	const int S = ( int )std::lround( std::sqrt( ( double )pixels.size() / 3.0 ) );
	std::printf( "pixel_values: %zu floats -> image size %d\n", pixels.size(), S );

	std::string			 err;
	trellis2_dino_model* m = trellis2_dino_load( gguf_path, true, &err );
	if( !m ) {
		std::fprintf( stderr, "load failed: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "backend: %s\n", trellis2_dino_backend_name( m ) );

	trellis2_dino_cond cond;
	trellis2_dino_taps taps;
	if( !trellis2_dino_encode( m, pixels.data(), S, cond, &taps, &err ) ) {
		std::fprintf( stderr, "encode failed: %s\n", err.c_str() );
		trellis2_dino_free( m );
		return 1;
	}

	int				   n_fail = 0, n_cmp = 0, n_missing = 0;
	std::string		   first_fail;
	std::vector<float> refbuf;
	for( size_t i = 0; i < taps.names.size(); ++i ) {
		const std::string& name = taps.names[i];
		if( !ref.has( name ) ) {
			++n_missing;
			continue;
		}
		if( !ref.load( name, refbuf ) ) {
			std::printf( "[%-18s] reference tensor unreadable -> FAIL\n", name.c_str() );
			++n_fail;
			continue;
		}
		++n_cmp;
		if( !t2_parity::compare( taps.data[i], refbuf, name, atol, rtol ) ) {
			++n_fail;
			if( first_fail.empty() )
				first_fail = name;
		}
	}

	trellis2_dino_free( m );

	std::printf( "\ncompared %d taps (%d without reference), %d failed\n", n_cmp, n_missing, n_fail );
	if( n_fail ) {
		std::printf( "FIRST DIVERGENCE: %s\n", first_fail.c_str() );
		std::printf( "RESULT: FAIL\n" );
		return 1;
	}
	if( n_cmp == 0 ) {
		std::printf( "RESULT: FAIL (nothing compared)\n" );
		return 1;
	}
	std::printf( "RESULT: PASS\n" );
	return 0;
}
