// Byte-exactness test of trellis2_preprocess_rgba against the Python
// reference (PIL): dumps/fixture_rgba.png -> preprocess -> must equal
// dumps/fixture_512.png (produced by scripts/dump_dino_reference.py).
//
// usage: test_preprocess <fixture_rgba.png> <fixture_512.png>
// exits 77 (ctest SKIP) when inputs are missing.

#include "trellis2.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <fixture_rgba.png> <fixture_512.png>\n", argv[0] );
		return 2;
	}
	if( !std::ifstream( argv[1] ).good() || !std::ifstream( argv[2] ).good() ) {
		std::fprintf( stderr, "missing input file(s), skipping\n" );
		return 77;
	}

	int			   w = 0, h = 0, comp = 0;
	unsigned char* rgba = stbi_load( argv[1], &w, &h, &comp, 4 );
	if( !rgba ) {
		std::fprintf( stderr, "decode failed: %s\n", stbi_failure_reason() );
		return 1;
	}

	int			   rw = 0, rh = 0, rc = 0;
	unsigned char* ref = stbi_load( argv[2], &rw, &rh, &rc, 3 );
	if( !ref ) {
		std::fprintf( stderr, "decode failed: %s\n", stbi_failure_reason() );
		stbi_image_free( rgba );
		return 1;
	}

	std::string			 err;
	std::vector<uint8_t> got;
	if( !trellis2_preprocess_rgba( rgba, w, h, rw, got, &err ) ) {
		std::fprintf( stderr, "preprocess failed: %s\n", err.c_str() );
		return 1;
	}

	const size_t n		= ( size_t )rw * rh * 3;
	size_t		 n_diff = 0, max_diff = 0, first = 0;
	for( size_t i = 0; i < n; ++i ) {
		const int d = std::abs( ( int )got[i] - ( int )ref[i] );
		if( d ) {
			if( !n_diff )
				first = i;
			++n_diff;
			if( ( size_t )d > max_diff )
				max_diff = ( size_t )d;
		}
	}

	std::printf( "preprocess %dx%d -> %dx%d: %zu/%zu bytes differ (max %zu)\n", w, h, rw, rh, n_diff, n, max_diff );
	stbi_image_free( rgba );
	stbi_image_free( ref );

	// Byte-exact is the goal; tolerate nothing so regressions are loud.
	if( n_diff != 0 ) {
		std::printf( "first diff at byte %zu\nRESULT: FAIL\n", first );
		return 1;
	}
	std::printf( "RESULT: PASS\n" );
	return 0;
}
