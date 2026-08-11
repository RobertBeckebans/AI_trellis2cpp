// dino_info — load a .dinodata DINOv3 conditioning file and print its shape,
// token breakdown, and fingerprints. The fingerprints should match the values
// in the matching `<stem>.dino.txt` JSON sidecar bit-for-bit.
//
//   usage: dino_info <path-to.dinodata>

#include "trellis2.h"

#include <cstdio>
#include <string>

int main( int argc, char** argv )
{
	if( argc < 2 ) {
		std::fprintf( stderr, "usage: %s <path-to.dinodata>\n", argv[0] );
		return 2;
	}

	std::printf( "trellis2.cpp %s\n", trellis2_version() );

	const std::string  path = argv[1];
	trellis2_dino_cond cond;
	std::string		   err;
	if( !trellis2_load_dinodata( path, cond, &err ) ) {
		std::fprintf( stderr, "error: %s\n", err.c_str() );
		return 1;
	}

	std::printf( "file            : %s\n", path.c_str() );
	std::printf( "format version  : %u\n", cond.format_version );

	std::printf( "shape           : [" );
	for( size_t i = 0; i < cond.shape.size(); ++i ) {
		std::printf( "%lld%s", ( long long )cond.shape[i], i + 1 < cond.shape.size() ? ", " : "" );
	}
	std::printf( "]\n" );

	const long long tok = ( long long )cond.tokens();
	std::printf( "tokens          : %lld (cls:1 + register:4 + patch:%lld)\n", tok, tok >= 5 ? tok - 5 : 0 );
	std::printf( "channels        : %lld\n", ( long long )cond.channels() );
	std::printf( "count           : %zu floats\n", cond.count() );

	const trellis2_dino_fingerprint fp = trellis2_dino_fingerprints( cond );
	std::printf( "fingerprints:\n" );
	std::printf( "  min   : %.6f\n", fp.vmin );
	std::printf( "  max   : %.6f\n", fp.vmax );
	std::printf( "  mean  : %.9f\n", fp.mean );
	std::printf( "  sum   : %.6f\n", fp.sum );
	std::printf( "  l2    : %.6f\n", fp.l2 );
	std::printf( "  count : %zu\n", fp.count );

	return 0;
}
