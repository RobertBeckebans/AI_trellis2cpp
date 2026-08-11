// ss_mesh — run the stage-1 SS decoder and export the occupancy isosurface as
// a Wavefront OBJ via (tetrahedral) marching cubes.
//
//   usage: ss_mesh <ss_dec.gguf> <z_s.latent> [out.obj] [--iso V] [--normalize]
//
// Decodes z_s -> a 64^3 occupancy logit grid, extracts the {logit = iso} surface
// (default iso 0, i.e. the occupancy boundary), and writes it as out.obj
// (default ss_mesh.obj). With --normalize, vertices are mapped from grid-index
// units into the centered unit cube [-0.5, 0.5]^3.

#include "trellis2.h"
#include "marching_cubes.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <ss_dec.gguf> <z_s.latent> [out.obj] [--iso V] [--normalize]\n", argv[0] );
		return 2;
	}
	const std::string gguf_path = argv[1];
	const std::string lat_path	= argv[2];
	std::string		  out_path	= "ss_mesh.obj";
	float			  iso		= 0.0f;
	bool			  normalize = false;
	bool			  out_set	= false;
	for( int i = 3; i < argc; ++i ) {
		if( std::strcmp( argv[i], "--iso" ) == 0 && i + 1 < argc )
			iso = ( float )std::atof( argv[++i] );
		else if( std::strcmp( argv[i], "--normalize" ) == 0 )
			normalize = true;
		else if( !out_set ) {
			out_path = argv[i];
			out_set	 = true;
		}
	}

	std::printf( "trellis2.cpp %s\n", trellis2_version() );

	std::string			   err;
	trellis2_ss_dec_model* m = trellis2_ss_dec_load( gguf_path, true, &err );
	if( !m ) {
		std::fprintf( stderr, "model load error: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "backend: %s\n", trellis2_ss_dec_backend_name( m ) );

	const trellis2_ss_dec_hparams hp	= trellis2_ss_dec_hparams_of( m );
	const int					  Rin	= hp.res_in();
	const int					  Rout	= hp.res_out();
	const size_t				  n_in	= ( size_t )hp.latent_channels * Rin * Rin * Rin;
	const size_t				  n_out = ( size_t )hp.out_channels * Rout * Rout * Rout;

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

	std::vector<float> logits( n_out, 0.0f );
	if( !trellis2_ss_dec_decode( m, latent.data(), logits.data(), &err ) ) {
		std::fprintf( stderr, "decode error: %s\n", err.c_str() );
		trellis2_ss_dec_free( m );
		return 1;
	}
	trellis2_ss_dec_free( m );

	size_t occ = 0;
	for( float v : logits )
		if( v > iso )
			++occ;
	std::printf( "logits : [%d^3]  occupied(>%.2f)=%zu/%zu (%.2f%%)\n", Rout, iso, occ, n_out, 100.0 * ( double )occ / ( double )n_out );

	// Decoder output is channel-major [1, R, R, R] with linear index
	// i*R^2 + j*R + k (k fastest) -> matches marching_cubes' x + y*R + z*R^2 with
	// (x,y,z) = (k, j, i).
	mc::Mesh mesh = mc::extract( logits.data(), Rout, Rout, Rout, iso );
	std::printf( "mesh   : %zu verts, %zu tris\n", mesh.n_verts(), mesh.n_tris() );

	if( mesh.n_tris() == 0 ) {
		std::fprintf( stderr, "warning: empty surface at iso=%.3f (nothing to write)\n", iso );
		return 1;
	}

	if( normalize ) {
		for( size_t i = 0; i < mesh.verts.size(); ++i )
			mesh.verts[i] = mesh.verts[i] / ( float )Rout - 0.5f;
	}

	if( !mc::write_obj( mesh, out_path.c_str() ) ) {
		std::fprintf( stderr, "failed to write %s\n", out_path.c_str() );
		return 1;
	}
	std::printf( "wrote %s\n", out_path.c_str() );
	return 0;
}
