// libFuzzer harness for the .dinodata loader: arbitrary bytes -> temp file ->
// trellis2_load_dinodata -> (on success) fingerprints over the payload.
//
// .dinodata files typically come from this codebase itself, but the demo and
// CLI accept arbitrary paths, so the parser must be robust: reject bad magic /
// truncated headers / shape overflow cleanly, never over-read.
//
// Invariants checked with abort() (not just "no crash"):
//   * on success, data.size() == product(shape) and shape is non-empty
//   * fingerprints run over the whole payload without UB
//
// Run: ./build-fuzz/fuzz/fuzz_dinodata corpus_dinodata/ -max_len=1048576

#include "../trellis2.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

extern "C" int LLVMFuzzerTestOneInput( const uint8_t* data, size_t size )
{
	if( size > ( 16u << 20 ) ) {
		return 0;
	}

	char	  path[] = "/tmp/fuzz_dinodata_XXXXXX";
	const int fd	 = mkstemp( path );
	if( fd < 0 ) {
		return 0;
	}
	const ssize_t written = write( fd, data, size );
	close( fd );
	if( written != ( ssize_t )size ) {
		unlink( path );
		return 0;
	}

	trellis2_dino_cond cond;
	std::string		   err;
	if( trellis2_load_dinodata( path, cond, &err ) ) {
		int64_t total = 1;
		for( int64_t d : cond.shape )
			total *= d;
		if( cond.shape.empty() || ( size_t )total != cond.data.size() ) {
			abort(); // loader produced an inconsistent cond
		}
		const trellis2_dino_fingerprint fp = trellis2_dino_fingerprints( cond );
		if( fp.count != cond.data.size() ) {
			abort();
		}
	} else if( err.empty() ) {
		abort(); // failure must come with a reason
	}

	unlink( path );
	return 0;
}
