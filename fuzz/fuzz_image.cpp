// libFuzzer harness for the image upload path: arbitrary bytes ->
// t2_preprocess_image_bytes (stb_image decode -> alpha bbox crop ->
// premultiply -> PIL-compatible Lanczos resize).
//
// This is the demo server's untrusted-input surface: the Go server hands
// uploaded bytes straight to this function. GGUF model files are trusted
// assets and deliberately not fuzzed here (privacy-filter.cpp threat model).
//
// Build via -DTRELLIS2_FUZZ=ON (clang only); run:
//   ./build-fuzz/fuzz/fuzz_image corpus_image/ -max_len=1048576
//
// Success invariant: the function either fails cleanly (nonzero + error
// message) or fills exactly out_size^2*3 bytes. Any crash/leak/UB is a bug.

#include "../trellis2_capi.h"

#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput( const uint8_t* data, size_t size )
{
	if( size == 0 || size > ( 32u << 20 ) ) {
		return 0;
	}

	// Small output target keeps per-input cost low; the resampler code path
	// is identical for any out_size.
	static thread_local std::vector<unsigned char> out( 64 * 64 * 3 );
	char										   err[256] = { 0 };

	( void )t2_preprocess_image_bytes( data, ( int )size, 64, out.data(), err, sizeof( err ) );
	return 0;
}
