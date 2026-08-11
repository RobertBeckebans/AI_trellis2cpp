// ss_flow_info — load a converted SS-flow DiT GGUF and print its hyperparameters
// and tensor inventory. Validates that convert_ss_flow_to_gguf.py produced a file
// ggml can read, and that every weight is reachable by name.
//
//   usage: ss_flow_info <path-to.gguf> [--load]
//
// By default only metadata is parsed (fast). Pass --load to also read all weight
// data into host memory (~2.6 GB for the f16 checkpoint).

#include "trellis2.h"

#include <cstdio>
#include <cstring>
#include <string>

int main( int argc, char** argv )
{
	if( argc < 2 ) {
		std::fprintf( stderr, "usage: %s <path-to.gguf> [--load]\n", argv[0] );
		return 2;
	}
	const std::string path		   = argv[1];
	const bool		  load_tensors = ( argc > 2 && std::strcmp( argv[2], "--load" ) == 0 );

	std::printf( "trellis2.cpp %s\n", trellis2_version() );

	std::string				err;
	trellis2_ss_flow_model* m = trellis2_ss_flow_load( path, load_tensors, &err );
	if( !m ) {
		std::fprintf( stderr, "error: %s\n", err.c_str() );
		return 1;
	}

	const trellis2_ss_flow_hparams& hp = trellis2_ss_flow_hparams_of( m );
	std::printf( "file            : %s  (%s)\n", path.c_str(), load_tensors ? "weights loaded" : "metadata only" );
	if( load_tensors ) {
		std::printf( "backend         : %s\n", trellis2_ss_flow_backend_name( m ) );
	}
	std::printf( "hyperparameters:\n" );
	std::printf( "  resolution      : %d (grid %d^3 = %d tokens)\n", hp.resolution, hp.resolution, hp.resolution * hp.resolution * hp.resolution );
	std::printf( "  in/out channels : %d / %d\n", hp.in_channels, hp.out_channels );
	std::printf( "  model_channels  : %d\n", hp.model_channels );
	std::printf( "  cond_channels   : %d\n", hp.cond_channels );
	std::printf( "  num_blocks      : %d\n", hp.num_blocks );
	std::printf( "  num_heads       : %d (head_dim %d)\n", hp.num_heads, hp.head_dim() );
	std::printf( "  mlp_ratio       : %.4f\n", hp.mlp_ratio );
	std::printf( "  pe_mode         : %s (freq %.1f..%.1f)\n", hp.pe_mode, hp.rope_freq_min, hp.rope_freq_base );
	std::printf( "  share_mod       : %s\n", hp.share_mod ? "true" : "false" );
	std::printf( "  qk_rms_norm     : self=%s cross=%s\n", hp.qk_rms_norm ? "true" : "false", hp.qk_rms_norm_cross ? "true" : "false" );
	std::printf( "  file_type       : %d (0=f32,1=f16,2=bf16)\n", hp.file_type );

	const int n = trellis2_ss_flow_n_tensors( m );
	std::printf( "tensors         : %d\n", n );

	// Print the global (non-block) tensors in full, then summarize block 0.
	std::printf( "  -- global + block 0 (sample) --\n" );
	size_t total_bytes = 0;
	int	   shown	   = 0;
	for( int i = 0; i < n; ++i ) {
		trellis2_tensor_info ti;
		if( !trellis2_ss_flow_get_tensor_info( m, i, ti ) )
			continue;
		total_bytes += ti.n_bytes;
		const bool is_global = ti.name.rfind( "blocks.", 0 ) != 0;
		const bool is_block0 = ti.name.rfind( "blocks.0.", 0 ) == 0;
		if( ( is_global || is_block0 ) && shown < 40 ) {
			std::printf( "    %-40s %-4s [", ti.name.c_str(), ti.type_name.c_str() );
			for( int d = ti.n_dims - 1; d >= 0; --d ) {
				std::printf( "%lld%s", ( long long )ti.ne[d], d ? ", " : "" );
			}
			std::printf( "]\n" );
			++shown;
		}
	}
	std::printf( "total weight bytes: %.2f MB\n", total_bytes / ( 1024.0 * 1024.0 ) );

	// Spot-check a couple of expected names.
	const char* probes[] = {
		"input_layer.weight",
		"adaLN_modulation.1.weight",
		"blocks.0.self_attn.to_qkv.weight",
		"blocks.29.cross_attn.to_kv.weight",
		"out_layer.weight",
	};
	std::printf( "name probes:\n" );
	for( const char* p : probes ) {
		std::printf( "  %-40s %s\n", p, trellis2_ss_flow_has_tensor( m, p ) ? "present" : "MISSING" );
	}

	trellis2_ss_flow_free( m );
	return 0;
}
