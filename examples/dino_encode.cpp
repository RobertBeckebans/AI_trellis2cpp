// Encode an image into the TRELLIS.2 DINOv3 conditioning tensor (.dinodata),
// replacing the external dump_dinodata.py:
//
//   image (PNG/JPG; solid black/white backgrounds are removed automatically)
//     -> background cleanup -> alpha bbox crop, premultiply, LANCZOS 512
//     -> DINOv3 ViT-L/16 -> [1, 1029, 1024] cond -> .dinodata
//
// usage: dino_encode <dino.gguf> <image> [out.dinodata] [--size N] [--pre out.png]
//
#include "trellis2.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main( int argc, char** argv )
{
	if( argc < 3 ) {
		std::fprintf( stderr, "usage: %s <dino.gguf> <image> [out.dinodata] [--size N] [--pre out.png]\n", argv[0] );
		return 2;
	}
	const std::string gguf_path = argv[1];
	const std::string img_path	= argv[2];
	std::string		  out_path	= "cond.dinodata";
	std::string		  pre_path;
	int				  size = 512;
	for( int i = 3; i < argc; ++i ) {
		if( std::strcmp( argv[i], "--size" ) == 0 && i + 1 < argc ) {
			size = std::atoi( argv[++i] );
		} else if( std::strcmp( argv[i], "--pre" ) == 0 && i + 1 < argc ) {
			pre_path = argv[++i];
		} else {
			out_path = argv[i];
		}
	}

	int			   w = 0, h = 0, comp = 0;
	unsigned char* pixels = stbi_load( img_path.c_str(), &w, &h, &comp, 4 );
	if( !pixels ) {
		std::fprintf( stderr, "failed to decode image %s: %s\n", img_path.c_str(), stbi_failure_reason() );
		return 1;
	}
	const int removed = trellis2_remove_solid_background_rgba( pixels, w, h, TRELLIS2_BACKGROUND_AUTO );
	std::printf( "image  : %s  %dx%d (%d channels, background pixels changed: %d)\n", img_path.c_str(), w, h, comp, removed );

	std::string			 err;
	std::vector<uint8_t> rgb;
	if( !trellis2_preprocess_rgba( pixels, w, h, size, rgb, &err ) ) {
		std::fprintf( stderr, "preprocess failed: %s\n", err.c_str() );
		stbi_image_free( pixels );
		return 1;
	}
	stbi_image_free( pixels );

	if( !pre_path.empty() ) {
		stbi_write_png( pre_path.c_str(), size, size, 3, rgb.data(), size * 3 );
		std::printf( "wrote  : %s (preprocessed %dx%d RGB)\n", pre_path.c_str(), size, size );
	}

	trellis2_dino_model* model = trellis2_dino_load( gguf_path, true, &err );
	if( !model ) {
		std::fprintf( stderr, "model load failed: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "model  : %s (backend %s)\n", gguf_path.c_str(), trellis2_dino_backend_name( model ) );

	trellis2_dino_cond cond;
	if( !trellis2_dino_encode_rgb( model, rgb.data(), size, cond, &err ) ) {
		std::fprintf( stderr, "encode failed: %s\n", err.c_str() );
		trellis2_dino_free( model );
		return 1;
	}
	trellis2_dino_free( model );

	const trellis2_dino_fingerprint fp = trellis2_dino_fingerprints( cond );
	std::printf( "cond   : [1, %lld, %lld]  min=%.4f max=%.4f mean=%.6f l2=%.4f\n", ( long long )cond.tokens(), ( long long )cond.channels(), fp.vmin, fp.vmax, fp.mean, fp.l2 );

	if( !trellis2_save_dinodata( out_path, cond, &err ) ) {
		std::fprintf( stderr, "save failed: %s\n", err.c_str() );
		return 1;
	}
	std::printf( "wrote  : %s (%zu floats)\n", out_path.c_str(), cond.count() );
	return 0;
}
