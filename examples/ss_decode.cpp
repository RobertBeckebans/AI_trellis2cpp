// ss_decode — run the stage-1 SS decoder: a sparse-structure latent z_s
// (.latent from ss_sample) + the SS decoder GGUF -> an occupancy logit grid.
// The coarse voxel scaffold is logit > 0.
//
//   usage: ss_decode <ss_dec.gguf> <z_s.latent> [out.occ]
//
// Writes the occupancy logit grid (channel-major [out_channels * R_out^3]
// float32) to out.occ if given. Also prints occupancy stats and, if all output
// channels are 1, the count of occupied voxels.

#include "trellis2.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <ss_dec.gguf> <z_s.latent> [out.occ]\n", argv[0] );
		return 2;
	}
	const std::string gguf_path = argv[1];
	const std::string lat_path	= argv[2];
	const std::string out_path	= argc > 3 ? argv[3] : "";

	std::printf( "trellis2.cpp %s\n", trellis2_version() );

	std::string			   err;
	trellis2_ss_dec_model* m = trellis2_ss_dec_load( gguf_path, true, &err );
	if( !m ) {
		std::fprintf( stderr, "model load error: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "backend: %s\n", trellis2_ss_dec_backend_name( m ) );

	const trellis2_ss_dec_hparams hp	= trellis2_ss_dec_hparams_of( m ); // copy (used after free)
	const int					  Rin	= hp.res_in();
	const int					  Rout	= hp.res_out();
	const size_t				  n_in	= ( size_t )hp.latent_channels * Rin * Rin * Rin;
	const size_t				  n_out = ( size_t )hp.out_channels * Rout * Rout * Rout;

	// load latent
	std::ifstream				  f( lat_path, std::ios::binary );
	if( !f ) {
		std::fprintf( stderr, "cannot open latent %s\n", lat_path.c_str() );
		trellis2_ss_dec_free( m );
		return 1;
	}
	std::vector<float> latent( n_in );
	f.read( reinterpret_cast<char*>( latent.data() ), ( std::streamsize )( n_in * sizeof( float ) ) );
	if( ( size_t )f.gcount() != n_in * sizeof( float ) ) {
		std::fprintf( stderr, "latent size mismatch: got %zu floats, want %zu\n", ( size_t )f.gcount() / sizeof( float ), n_in );
		trellis2_ss_dec_free( m );
		return 1;
	}
	std::printf( "latent : [%d,%d,%d,%d]\n", hp.latent_channels, Rin, Rin, Rin );

	std::vector<float> occ( n_out, 0.0f );
	if( !trellis2_ss_dec_decode( m, latent.data(), occ.data(), &err ) ) {
		std::fprintf( stderr, "decode error: %s\n", err.c_str() );
		trellis2_ss_dec_free( m );
		return 1;
	}
	trellis2_ss_dec_free( m );

	double mn = 1e30, mx = -1e30, sum = 0.0;
	size_t occupied = 0;
	for( float v : occ ) {
		mn = v < mn ? v : mn;
		mx = v > mx ? v : mx;
		sum += v;
		if( v > 0.0f )
			++occupied;
	}
	std::printf( "logits : [%d,%d,%d,%d] min=%.4f max=%.4f mean=%.5f  occupied(>0)=%zu/%zu (%.2f%%)\n",
		hp.out_channels,
		Rout,
		Rout,
		Rout,
		mn,
		mx,
		sum / ( double )n_out,
		occupied,
		n_out,
		100.0 * ( double )occupied / ( double )n_out );

	if( !out_path.empty() ) {
		std::ofstream o( out_path, std::ios::binary );
		o.write( reinterpret_cast<const char*>( occ.data() ), ( std::streamsize )( n_out * sizeof( float ) ) );
		std::printf( "wrote %s (%zu floats)\n", out_path.c_str(), n_out );
	}
	return 0;
}
