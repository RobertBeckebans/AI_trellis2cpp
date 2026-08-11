// ss_sample — run stage-1 sampling: a DINOv3 .dinodata cond + the SS-flow DiT
// GGUF -> the sparse-structure latent z_s. The occupancy scaffold is z_s > 0.
//
//   usage: ss_sample <ss_flow_dit.gguf> <cond.dinodata> [out.latent] [--seed N]
//
// Writes the z_s latent (channel-major [in_channels * R^3] float32) to out.latent
// if given — the input the SS decoder (next stage) will consume.

#include "trellis2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <ss_flow_dit.gguf> <cond.dinodata> [out.latent] [--seed N]\n", argv[0] );
		return 2;
	}
	const std::string gguf_path = argv[1];
	const std::string dino_path = argv[2];
	std::string		  out_path;
	uint64_t		  seed = 0;
	for( int i = 3; i < argc; ++i ) {
		if( std::strcmp( argv[i], "--seed" ) == 0 && i + 1 < argc )
			seed = std::strtoull( argv[++i], nullptr, 10 );
		else if( out_path.empty() )
			out_path = argv[i];
	}

	std::printf( "trellis2.cpp %s\n", trellis2_version() );

	trellis2_dino_cond cond;
	std::string		   err;
	if( !trellis2_load_dinodata( dino_path, cond, &err ) ) {
		std::fprintf( stderr, "cond load error: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "cond  : %lld tokens x %lld channels\n", ( long long )cond.tokens(), ( long long )cond.channels() );

	trellis2_ss_flow_model* m = trellis2_ss_flow_load( gguf_path, true, &err );
	if( !m ) {
		std::fprintf( stderr, "model load error: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "backend: %s\n", trellis2_ss_flow_backend_name( m ) );

	const trellis2_ss_flow_hparams hp = trellis2_ss_flow_hparams_of( m ); // copy (used after free)
	const size_t				   N  = ( size_t )hp.resolution * hp.resolution * hp.resolution;
	const size_t				   n  = ( size_t )hp.in_channels * N;

	trellis2_ss_sampler_params	   P; // pipeline defaults (12 steps, gs 7.5, ...)
	P.seed = seed;

	std::vector<float> latent( n, 0.0f );
	if( !trellis2_ss_flow_sample( m, cond.data.data(), ( int )cond.tokens(), ( int )cond.channels(), &P, /*noise*/ nullptr, latent.data(), &err ) ) {
		std::fprintf( stderr, "sample error: %s\n", err.c_str() );
		trellis2_ss_flow_free( m );
		return 1;
	}
	trellis2_ss_flow_free( m );

	double mn = 1e30, mx = -1e30, sum = 0.0;
	size_t occ = 0;
	for( float v : latent ) {
		mn = v < mn ? v : mn;
		mx = v > mx ? v : mx;
		sum += v;
		if( v > 0.0f )
			++occ;
	}
	std::printf( "z_s   : [%d,%d,%d,%d] min=%.4f max=%.4f mean=%.5f  occupancy(>0)=%.2f%%\n",
		hp.in_channels,
		hp.resolution,
		hp.resolution,
		hp.resolution,
		mn,
		mx,
		sum / ( double )n,
		100.0 * ( double )occ / ( double )n );

	if( !out_path.empty() ) {
		std::ofstream f( out_path, std::ios::binary );
		f.write( reinterpret_cast<const char*>( latent.data() ), ( std::streamsize )( n * sizeof( float ) ) );
		std::printf( "wrote %s (%zu floats)\n", out_path.c_str(), n );
	}
	return 0;
}
