#include "trellis2.h"

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <vector>
#include <algorithm>
#include <array>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <random>
#include <thread>
#include <unordered_map>

/*****************************************************************************
** Helpers
*****************************************************************************/

namespace
{

inline void set_error( std::string* error, const std::string& msg )
{
	if( error )
		*error = msg;
}

// Read a little-endian uint32 from a byte cursor, advancing it.
inline bool read_u32_le( const uint8_t*& p, const uint8_t* end, uint32_t& out )
{
	if( p + 4 > end )
		return false;
	out = ( uint32_t )p[0] | ( ( uint32_t )p[1] << 8 ) | ( ( uint32_t )p[2] << 16 ) | ( ( uint32_t )p[3] << 24 );
	p += 4;
	return true;
}

} // namespace

/*****************************************************************************
** Version
*****************************************************************************/

const char* trellis2_version()
{
	return TRELLIS2_VERSION;
}

/*****************************************************************************
** .dinodata loader
**
** Binary layout (little-endian), produced by dump_dinodata.py:
**   magic   : 8 bytes  "DINOCOND"
**   version : uint32
**   dtype   : uint32   (0 = float32)   -- only float32 is supported here
**   ndim    : uint32
**   shape   : ndim * uint32            (C-contiguous in this order)
**   payload : prod(shape) * float32    (little-endian)
*****************************************************************************/

bool trellis2_load_dinodata( const std::string& path, trellis2_dino_cond& out, std::string* error )
{
	out = trellis2_dino_cond {};

	std::ifstream f( path, std::ios::binary );
	if( !f ) {
		set_error( error, "cannot open file: " + path );
		return false;
	}

	// Slurp the whole file — these are a few MB, well within memory.
	std::vector<uint8_t> buf( ( std::istreambuf_iterator<char>( f ) ), std::istreambuf_iterator<char>() );
	if( buf.size() < 8 + 12 ) {
		set_error( error, "file too small to contain a .dinodata header" );
		return false;
	}

	const uint8_t*	  p	  = buf.data();
	const uint8_t*	  end = buf.data() + buf.size();

	static const char MAGIC[8] = { 'D', 'I', 'N', 'O', 'C', 'O', 'N', 'D' };
	if( std::memcmp( p, MAGIC, 8 ) != 0 ) {
		set_error( error, "bad magic (expected 'DINOCOND')" );
		return false;
	}
	p += 8;

	uint32_t version = 0, dtype = 0, ndim = 0;
	if( !read_u32_le( p, end, version ) || !read_u32_le( p, end, dtype ) || !read_u32_le( p, end, ndim ) ) {
		set_error( error, "truncated header" );
		return false;
	}

	if( dtype != 0 ) {
		set_error( error, "unsupported dtype " + std::to_string( dtype ) + " (only 0=float32 is supported)" );
		return false;
	}
	if( ndim == 0 || ndim > 8 ) {
		set_error( error, "implausible ndim " + std::to_string( ndim ) );
		return false;
	}

	// The shape product is attacker-controlled and must not wrap: cap the
	// element count well above any real conditioning tensor but far below
	// anything that could overflow want_bytes or exhaust memory.
	const int64_t		 MAX_ELEMS = ( int64_t )1 << 31;

	std::vector<int64_t> shape( ndim );
	int64_t				 total = 1;
	for( uint32_t i = 0; i < ndim; ++i ) {
		uint32_t dim = 0;
		if( !read_u32_le( p, end, dim ) ) {
			set_error( error, "truncated shape" );
			return false;
		}
		if( dim == 0 || ( int64_t )dim > MAX_ELEMS || total > MAX_ELEMS / ( int64_t )dim ) {
			set_error( error, "implausible shape (zero or overflowing element count)" );
			return false;
		}
		shape[i] = ( int64_t )dim;
		total *= ( int64_t )dim;
	}

	const size_t want_bytes = ( size_t )total * sizeof( float );
	const size_t have_bytes = ( size_t )( end - p );
	if( have_bytes < want_bytes ) {
		set_error( error, "payload truncated: have " + std::to_string( have_bytes ) + " bytes, need " + std::to_string( want_bytes ) );
		return false;
	}

	out.shape		   = std::move( shape );
	out.format_version = version;
	out.data.resize( ( size_t )total );
	// Little-endian float32 on the host (all targets we build for are LE).
	std::memcpy( out.data.data(), p, want_bytes );

	return true;
}

/*****************************************************************************
** Fingerprints
*****************************************************************************/

trellis2_dino_fingerprint trellis2_dino_fingerprints( const trellis2_dino_cond& cond )
{
	trellis2_dino_fingerprint fp;
	fp.count = cond.data.size();
	if( cond.data.empty() ) {
		return fp;
	}

	float  vmin	 = std::numeric_limits<float>::infinity();
	float  vmax	 = -std::numeric_limits<float>::infinity();
	double sum	 = 0.0;
	double sumsq = 0.0;
	for( float v : cond.data ) {
		if( v < vmin )
			vmin = v;
		if( v > vmax )
			vmax = v;
		sum += ( double )v;
		sumsq += ( double )v * ( double )v;
	}

	fp.vmin = vmin;
	fp.vmax = vmax;
	fp.sum	= sum;
	fp.mean = sum / ( double )fp.count;
	fp.l2	= std::sqrt( sumsq );
	return fp;
}

/*****************************************************************************
** Sparse-structure flow DiT (stage 1) — GGUF loader
*****************************************************************************/

struct trellis2_ss_flow_model {
	gguf_context*								  gguf = nullptr;
	ggml_context*								  ctx  = nullptr;
	trellis2_ss_flow_hparams					  hp;
	bool										  has_data = false; // true if weight payloads were read (load_tensors)

	// Compute backend (auto-selected: GPU if available, else CPU) and the
	// buffer holding the weights on that backend. Only set when has_data.
	ggml_backend_t								  backend	  = nullptr;
	ggml_backend_buffer_t						  weights_buf = nullptr;
	std::string									  backend_name;

	// name -> tensor (into ctx); built once at load for O(1) graph wiring later.
	std::unordered_map<std::string, ggml_tensor*> tensors;
};

namespace
{

// Pick the best available compute backend: the first GPU device exposed by the
// ggml backend registry (CUDA / Metal / Vulkan / ...), falling back to CPU.
// Mirrors sam3.cpp's "use a GPU backend automatically if one is available".
// device: nullptr/"auto" = GPU if available else CPU; "cpu" = force CPU.
// The TRELLIS2_DEVICE env var overrides "auto".
ggml_backend_t init_best_backend( std::string& name_out, const char* device = nullptr )
{
	std::string want = device ? device : "";
	if( want.empty() || want == "auto" ) {
		if( const char* env = std::getenv( "TRELLIS2_DEVICE" ) )
			want = env;
	}
	if( want != "cpu" )
		for( size_t i = 0; i < ggml_backend_dev_count(); ++i ) {
			ggml_backend_dev_t dev = ggml_backend_dev_get( i );
			if( ggml_backend_dev_type( dev ) == GGML_BACKEND_DEVICE_TYPE_GPU ) {
				ggml_backend_t b = ggml_backend_dev_init( dev, nullptr );
				if( b ) {
					const char* d = ggml_backend_dev_description( dev );
					name_out	  = d ? d : ggml_backend_dev_name( dev );
					return b;
				}
			}
		}
	name_out		   = "CPU";
	ggml_backend_t cpu = ggml_backend_cpu_init();
	// ggml defaults to 4 threads; use every core (TRELLIS2_N_THREADS overrides).
	int			   n_threads = ( int )std::thread::hardware_concurrency();
	if( const char* env = std::getenv( "TRELLIS2_N_THREADS" ) ) {
		const int v = std::atoi( env );
		if( v > 0 )
			n_threads = v;
	}
	if( n_threads > 0 )
		ggml_backend_cpu_set_n_threads( cpu, n_threads );
	return cpu;
}

// KV readers with defaults (return the default if the key is absent).
uint32_t kv_u32( const gguf_context* g, const char* key, uint32_t def )
{
	const int64_t id = gguf_find_key( g, key );
	return id < 0 ? def : gguf_get_val_u32( g, id );
}
float kv_f32( const gguf_context* g, const char* key, float def )
{
	const int64_t id = gguf_find_key( g, key );
	return id < 0 ? def : gguf_get_val_f32( g, id );
}
bool kv_bool( const gguf_context* g, const char* key, bool def )
{
	const int64_t id = gguf_find_key( g, key );
	return id < 0 ? def : gguf_get_val_bool( g, id );
}
const char* kv_str( const gguf_context* g, const char* key, const char* def )
{
	const int64_t id = gguf_find_key( g, key );
	return id < 0 ? def : gguf_get_val_str( g, id );
}

} // namespace

size_t trellis2_gpu_free_vram()
{
	ggml_backend_dev_t dev = ggml_backend_dev_by_type( GGML_BACKEND_DEVICE_TYPE_GPU );
	if( !dev )
		return 0; // CPU-only build/host
	size_t free = 0, total = 0;
	ggml_backend_dev_memory( dev, &free, &total );
	return free;
}

trellis2_ss_flow_model* trellis2_ss_flow_load( const std::string& path, bool load_tensors, std::string* error, const char* device )
{
	auto*			 m = new trellis2_ss_flow_model();

	// Always parse metadata only; the weights are then allocated on the chosen
	// backend and the payloads streamed in from the file (so the GPU can use
	// them directly). This is the standard llama.cpp / stable-diffusion.cpp path.
	gguf_init_params params;
	params.no_alloc = true;
	params.ctx		= &m->ctx;

	m->gguf = gguf_init_from_file( path.c_str(), params );
	if( !m->gguf ) {
		set_error( error, "gguf_init_from_file failed (not a GGUF file?): " + path );
		delete m;
		return nullptr;
	}

	// Sanity-check the architecture tag.
	const char* arch = kv_str( m->gguf, "general.architecture", "" );
	if( std::strcmp( arch, "trellis2-ss-flow" ) != 0 ) {
		set_error( error, std::string( "unexpected architecture '" ) + arch + "' (expected 'trellis2-ss-flow')" );
		trellis2_ss_flow_free( m );
		return nullptr;
	}

	trellis2_ss_flow_hparams& hp = m->hp;
	const char*				  P	 = "trellis2.ss_flow.";
	auto					  K	 = [&]( const char* suffix ) { return std::string( P ) + suffix; };

	hp.resolution		 = ( int32_t )kv_u32( m->gguf, K( "resolution" ).c_str(), 0 );
	hp.in_channels		 = ( int32_t )kv_u32( m->gguf, K( "in_channels" ).c_str(), 0 );
	hp.out_channels		 = ( int32_t )kv_u32( m->gguf, K( "out_channels" ).c_str(), 0 );
	hp.model_channels	 = ( int32_t )kv_u32( m->gguf, K( "model_channels" ).c_str(), 0 );
	hp.cond_channels	 = ( int32_t )kv_u32( m->gguf, K( "cond_channels" ).c_str(), 0 );
	hp.num_blocks		 = ( int32_t )kv_u32( m->gguf, K( "num_blocks" ).c_str(), 0 );
	hp.num_heads		 = ( int32_t )kv_u32( m->gguf, K( "num_heads" ).c_str(), 0 );
	hp.mlp_ratio		 = kv_f32( m->gguf, K( "mlp_ratio" ).c_str(), 0.0f );
	hp.share_mod		 = kv_bool( m->gguf, K( "share_mod" ).c_str(), false ) ? 1 : 0;
	hp.qk_rms_norm		 = kv_bool( m->gguf, K( "qk_rms_norm" ).c_str(), false ) ? 1 : 0;
	hp.qk_rms_norm_cross = kv_bool( m->gguf, K( "qk_rms_norm_cross" ).c_str(), false ) ? 1 : 0;
	hp.rope_freq_min	 = kv_f32( m->gguf, K( "rope_freq_min" ).c_str(), 1.0f );
	hp.rope_freq_base	 = kv_f32( m->gguf, K( "rope_freq_base" ).c_str(), 10000.0f );
	hp.file_type		 = ( int32_t )kv_u32( m->gguf, "general.file_type", 0 );
	std::snprintf( hp.pe_mode, sizeof( hp.pe_mode ), "%s", kv_str( m->gguf, K( "pe_mode" ).c_str(), "rope" ) );

	// Build name -> tensor map.
	for( ggml_tensor* t = ggml_get_first_tensor( m->ctx ); t != nullptr; t = ggml_get_next_tensor( m->ctx, t ) ) {
		m->tensors[t->name] = t;
	}

	if( load_tensors ) {
		// Allocate all weights on the auto-selected backend, then stream the
		// payloads from the file into that buffer.
		m->backend	   = init_best_backend( m->backend_name, device );
		m->weights_buf = ggml_backend_alloc_ctx_tensors( m->ctx, m->backend );
		if( !m->weights_buf ) {
			set_error( error, "failed to allocate weights on backend " + m->backend_name );
			trellis2_ss_flow_free( m );
			return nullptr;
		}

		std::ifstream fin( path, std::ios::binary );
		if( !fin ) {
			set_error( error, "cannot reopen file for weight data: " + path );
			trellis2_ss_flow_free( m );
			return nullptr;
		}
		const size_t		 data_off = gguf_get_data_offset( m->gguf );
		const int64_t		 nt		  = gguf_get_n_tensors( m->gguf );
		std::vector<uint8_t> buf;
		for( int64_t i = 0; i < nt; ++i ) {
			const char*	 name = gguf_get_tensor_name( m->gguf, i );
			ggml_tensor* t	  = m->tensors[name];
			const size_t nb	  = ggml_nbytes( t );
			const size_t off  = data_off + gguf_get_tensor_offset( m->gguf, i );
			buf.resize( nb );
			fin.seekg( ( std::streamoff )off, std::ios::beg );
			if( !fin.read( reinterpret_cast<char*>( buf.data() ), ( std::streamsize )nb ) ) {
				set_error( error, std::string( "failed reading weight '" ) + name + "' from file" );
				trellis2_ss_flow_free( m );
				return nullptr;
			}
			ggml_backend_tensor_set( t, buf.data(), 0, nb );
		}
		m->has_data = true;
	}

	return m;
}

void trellis2_ss_flow_free( trellis2_ss_flow_model* m )
{
	if( !m )
		return;
	if( m->weights_buf )
		ggml_backend_buffer_free( m->weights_buf );
	if( m->backend )
		ggml_backend_free( m->backend );
	if( m->gguf )
		gguf_free( m->gguf );
	if( m->ctx )
		ggml_free( m->ctx );
	delete m;
}

const char* trellis2_ss_flow_backend_name( const trellis2_ss_flow_model* m )
{
	return ( m && !m->backend_name.empty() ) ? m->backend_name.c_str() : "none";
}

const trellis2_ss_flow_hparams& trellis2_ss_flow_hparams_of( const trellis2_ss_flow_model* m )
{
	return m->hp;
}

int trellis2_ss_flow_n_tensors( const trellis2_ss_flow_model* m )
{
	return m ? ( int )gguf_get_n_tensors( m->gguf ) : 0;
}

bool trellis2_ss_flow_get_tensor_info( const trellis2_ss_flow_model* m, int i, trellis2_tensor_info& out )
{
	if( !m || i < 0 || i >= ( int )gguf_get_n_tensors( m->gguf ) )
		return false;
	const char* name = gguf_get_tensor_name( m->gguf, i );
	out.name		 = name;

	ggml_tensor* t = ggml_get_tensor( m->ctx, name );
	if( !t )
		return false;
	out.n_dims = ggml_n_dims( t );
	for( int d = 0; d < 4; ++d )
		out.ne[d] = t->ne[d];
	out.ggml_type = ( int )t->type;
	out.type_name = ggml_type_name( t->type );
	out.n_bytes	  = ggml_nbytes( t );
	return true;
}

bool trellis2_ss_flow_has_tensor( const trellis2_ss_flow_model* m, const std::string& name )
{
	return m && m->tensors.find( name ) != m->tensors.end();
}

/*****************************************************************************
** Sparse-structure flow DiT — forward pass (CPU backend)
**
** Mirrors SparseStructureFlowModel.forward + ModulatedTransformerCrossBlock:
**   h = input_layer(x)                                  # [C, N]
**   t_emb = adaLN(SiLU stack)(timestep_embedding(t))    # [6C] shared modulation
**   for each of num_blocks cross-blocks:
**     (shift/scale/gate)_{msa,mlp} = modulation_b + t_emb
**     h += gate_msa * self_attn( modulate(LN0(h)) )     # RoPE + QK-RMSNorm
**     h += cross_attn( LN1_affine(h), cond )            # QK-RMSNorm, no RoPE
**     h += gate_mlp * mlp( modulate(LN2(h)) )           # GELU-tanh FFN
**   out = out_layer(LayerNorm(h))                       # [out_channels, N]
*****************************************************************************/

namespace
{

// Scaled dot-product attention via ggml_flash_attn_ext (tiled online softmax,
// O(L) memory). q3/k3/v3 are [head_dim, n_head, L]; returns [n_head*head_dim, L_q].
//
// Flash is the default for both flow DiTs: it is bit-faithful to full softmax on
// CPU with F32 accumulation (validated to ~1e-4 rel-L2, identical to the exact
// materialized path) but avoids the [L_k, L_q, heads] score matrix — which is
// both the memory wall (the HR cascade's ~49k voxels would need >100 GB) and,
// on GPU, ~30% of the forward's wall time (the softmax + the permute/cont copies
// around it). On the CUDA F16-MMA kernel flash costs ~3e-3 rel-L2 per forward,
// immaterial to the final mesh. Set TRELLIS2_SDPA_EXACT to force the old
// materialized path (e.g. to reproduce the tightest GPU numbers).
ggml_tensor* sdpa_auto( ggml_context* ctx, ggml_tensor* q3, ggml_tensor* k3, ggml_tensor* v3, int C, float scale )
{
	ggml_tensor*	  qp = ggml_cont( ctx, ggml_permute( ctx, q3, 0, 2, 1, 3 ) ); // [hd, Lq, H]
	ggml_tensor*	  kp = ggml_cont( ctx, ggml_permute( ctx, k3, 0, 2, 1, 3 ) ); // [hd, Lk, H]
	ggml_tensor*	  vp = ggml_cont( ctx, ggml_permute( ctx, v3, 0, 2, 1, 3 ) ); // [hd, Lk, H]

	static const bool exact = std::getenv( "TRELLIS2_SDPA_EXACT" ) != nullptr;
	if( !exact ) {
		ggml_tensor* o = ggml_flash_attn_ext( ctx, qp, kp, vp, nullptr, scale, 0.0f, 0.0f );
		ggml_flash_attn_ext_set_prec( o, GGML_PREC_F32 );
		return ggml_reshape_2d( ctx, o, C, o->ne[2] ); // [C, Lq]
	}
	ggml_tensor* sc = ggml_mul_mat( ctx, kp, qp ); // [Lk, Lq, H]
	sc				= ggml_soft_max_ext( ctx, sc, nullptr, scale, 0.0f );
	ggml_tensor* vt = ggml_cont( ctx, ggml_permute( ctx, vp, 1, 0, 2, 3 ) ); // [Lk, hd, H]
	ggml_tensor* o	= ggml_mul_mat( ctx, vt, sc );							 // [hd, Lq, H]
	o				= ggml_cont( ctx, ggml_permute( ctx, o, 0, 2, 1, 3 ) );	 // [hd, H, Lq]
	return ggml_reshape_2d( ctx, o, C, o->ne[2] );							 // [C, Lq]
}

// Sinusoidal timestep embedding (cos|sin), matching TimestepEmbedder.
std::vector<float> timestep_embedding( float t, int dim )
{
	std::vector<float> e( ( size_t )dim, 0.0f );
	const int		   half = dim / 2;
	for( int i = 0; i < half; ++i ) {
		const float freq = std::exp( -std::log( 10000.0f ) * ( float )i / ( float )half );
		const float arg	 = t * freq;
		e[i]			 = std::cos( arg );
		e[half + i]		 = std::sin( arg );
	}
	return e; // dim is even here (256) so no padding needed
}

// Precompute the interleaved 3D-RoPE cos/sin tables for an R^3 grid.
// Layout matches q reshaped to [head_dim, n_heads, N]: cos/sin are [head_dim, 1, N]
// with cos[n*head_dim + 2p] == cos[n*head_dim + 2p+1] == cos(theta_p(n)).
void rope_tables( int res, int head_dim, float freq_min, float freq_base, std::vector<float>& cos_t, std::vector<float>& sin_t )
{
	const int		   dim		= 3;				  // 3 spatial axes
	const int		   freq_dim = head_dim / 2 / dim; // 21 for head_dim 128
	const int		   N		= res * res * res;

	std::vector<float> freqs( ( size_t )freq_dim );
	for( int mi = 0; mi < freq_dim; ++mi ) {
		freqs[mi] = freq_min / std::pow( freq_base, ( float )mi / ( float )freq_dim );
	}

	cos_t.assign( ( size_t )head_dim * N, 1.0f );
	sin_t.assign( ( size_t )head_dim * N, 0.0f );
	const int pairs = head_dim / 2; // 64
	for( int n = 0; n < N; ++n ) {
		const int coord[3] = { n / ( res * res ), ( n / res ) % res, n % res };
		for( int p = 0; p < pairs; ++p ) {
			float theta = 0.0f;		   // p == 63 -> pad (theta 0)
			if( p < dim * freq_dim ) { // p in 0..62
				theta = ( float )coord[p / freq_dim] * freqs[p % freq_dim];
			}
			const size_t base = ( size_t )n * head_dim + ( size_t )2 * p;
			cos_t[base] = cos_t[base + 1] = std::cos( theta );
			sin_t[base] = sin_t[base + 1] = std::sin( theta );
		}
	}
}

} // namespace

bool trellis2_ss_flow_forward( trellis2_ss_flow_model* m, const float* x, float t, const float* cond, int cond_tokens, int cond_channels, float* out, std::string* error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}

	const trellis2_ss_flow_hparams& hp = m->hp;
	if( std::strcmp( hp.pe_mode, "rope" ) != 0 ) {
		set_error( error, "only pe_mode=rope is implemented" );
		return false;
	}
	if( !hp.share_mod ) {
		set_error( error, "only share_mod=true is implemented" );
		return false;
	}
	if( cond_channels != hp.cond_channels ) {
		set_error( error, "cond_channels mismatch" );
		return false;
	}

	const bool	t2_timing = std::getenv( "TRELLIS2_TIMING" ) != nullptr;
	auto		t_now	  = [] { return std::chrono::steady_clock::now(); };
	auto		t_ms	  = []( auto a, auto b ) { return std::chrono::duration<double, std::milli>( b - a ).count(); };
	auto		t_start	  = t_now();

	const int	C		   = hp.model_channels; // 1536
	const int	R		   = hp.resolution;		// 16
	const int	N		   = R * R * R;			// 4096 tokens
	const int	H		   = hp.num_heads;		// 12
	const int	hd		   = hp.head_dim();		// 128
	const int	Lkv		   = cond_tokens;		// 1029
	const float attn_scale = 1.0f / std::sqrt( ( float )hd );

	std::string missing;
	auto		W = [&]( const std::string& n ) -> ggml_tensor* {
		   auto it = m->tensors.find( n );
		   if( it == m->tensors.end() ) {
			   if( missing.empty() )
				   missing = n;
			   return nullptr;
		   }
		   return it->second;
	};

	// ── compute graph context (metadata only; gallocr allocates data) ────────
	const size_t	 mem = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom( 32768, false );
	ggml_init_params ip { mem, nullptr, /*no_alloc*/ true };
	ggml_context*	 ctx = ggml_init( ip );
	ggml_cgraph*	 gf	 = ggml_new_graph_custom( ctx, 32768, false );

	// ── input leaves ─────────────────────────────────────────────────────────
	ggml_tensor*	 x_t   = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, N, hp.in_channels ); // channel-major [N, Cin]
	ggml_tensor*	 temb  = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, 256 );
	ggml_tensor*	 cos_t = ggml_new_tensor_3d( ctx, GGML_TYPE_F32, hd, 1, N );
	ggml_tensor*	 sin_t = ggml_new_tensor_3d( ctx, GGML_TYPE_F32, hd, 1, N );
	ggml_tensor*	 cnd   = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, cond_channels, Lkv ); // [ctx_ch, Lkv]
	ggml_set_input( x_t );
	ggml_set_name( x_t, "x" );
	ggml_set_input( temb );
	ggml_set_name( temb, "temb" );
	ggml_set_input( cos_t );
	ggml_set_input( sin_t );
	ggml_set_input( cnd );

	auto lin = [&]( ggml_tensor* in, const std::string& pfx ) -> ggml_tensor* {
		ggml_tensor* y = ggml_mul_mat( ctx, W( pfx + ".weight" ), in );
		ggml_tensor* b = W( pfx + ".bias" );
		if( b )
			y = ggml_add( ctx, y, b );
		return y;
	};
	// h * (1 + scale) + shift, broadcasting the [C] vectors over tokens.
	auto modulate = [&]( ggml_tensor* h, ggml_tensor* scale, ggml_tensor* shift ) { return ggml_add( ctx, ggml_add( ctx, ggml_mul( ctx, h, scale ), h ), shift ); };
	// interleaved RoPE on a [hd, H, N] tensor using the cos/sin tables.
	auto rope = [&]( ggml_tensor* q3 ) -> ggml_tensor* {
		ggml_tensor* q4	  = ggml_reshape_4d( ctx, q3, 2, hd / 2, H, N );
		ggml_tensor* q0	  = ggml_cont( ctx, ggml_view_4d( ctx, q4, 1, hd / 2, H, N, q4->nb[1], q4->nb[2], q4->nb[3], 0 ) );
		ggml_tensor* q1	  = ggml_cont( ctx, ggml_view_4d( ctx, q4, 1, hd / 2, H, N, q4->nb[1], q4->nb[2], q4->nb[3], q4->nb[0] ) );
		ggml_tensor* swap = ggml_concat( ctx, ggml_neg( ctx, q1 ), q0, 0 ); // [2,hd/2,H,N]
		swap			  = ggml_reshape_3d( ctx, swap, hd, H, N );
		return ggml_add( ctx, ggml_mul( ctx, q3, cos_t ), ggml_mul( ctx, swap, sin_t ) );
	};
	// QK-RMSNorm: F.normalize(x)*gamma*sqrt(hd) == rms_norm(x)*gamma (sqrt cancels).
	auto		 qk_norm = [&]( ggml_tensor* v3, const std::string& gname ) { return ggml_mul( ctx, ggml_rms_norm( ctx, v3, 1e-12f ), W( gname ) ); };
	// scaled-dot-product attention (flash: O(L) memory); q3/k3/v3 are [hd,H,L].
	auto		 sdpa = [&]( ggml_tensor* q3, ggml_tensor* k3, ggml_tensor* v3 ) { return sdpa_auto( ctx, q3, k3, v3, C, attn_scale ); };

	const size_t es = sizeof( float );

	// ── stem: input projection (+ no additive PE in rope mode) ───────────────
	ggml_tensor* h = ggml_cont( ctx, ggml_transpose( ctx, x_t ) ); // [Cin, N]
	h			   = lin( h, "input_layer" );					   // [C, N]

	// ── shared modulation from the timestep ──────────────────────────────────
	ggml_tensor* te	  = lin( temb, "t_embedder.mlp.0" );
	te				  = ggml_silu( ctx, te );
	te				  = lin( te, "t_embedder.mlp.2" );					   // [C]
	ggml_tensor* tmod = lin( ggml_silu( ctx, te ), "adaLN_modulation.1" ); // [6C]

	ggml_tensor* cond_h = cnd; // [ctx_ch, Lkv]

	for( int b = 0; b < hp.num_blocks; ++b ) {
		const std::string blk		= "blocks." + std::to_string( b );
		ggml_tensor*	  mods		= ggml_add( ctx, W( blk + ".modulation" ), tmod ); // [6C]
		auto			  chunk		= [&]( int idx ) { return ggml_view_1d( ctx, mods, C, ( size_t )idx * C * es ); };
		ggml_tensor *	  shift_msa = chunk( 0 ), *scale_msa = chunk( 1 ), *gate_msa = chunk( 2 );
		ggml_tensor *	  shift_mlp = chunk( 3 ), *scale_mlp = chunk( 4 ), *gate_mlp = chunk( 5 );

		// self-attention (norm1 affine-free, modulated; RoPE + QK-RMSNorm)
		ggml_tensor*	  hn  = modulate( ggml_norm( ctx, h, 1e-6f ), scale_msa, shift_msa );
		ggml_tensor*	  qkv = lin( hn, blk + ".self_attn.to_qkv" ); // [3C, N]
		ggml_tensor*	  q	  = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, qkv, C, N, qkv->nb[1], 0 ) ), hd, H, N );
		ggml_tensor*	  k	  = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, qkv, C, N, qkv->nb[1], ( size_t )C * es ) ), hd, H, N );
		ggml_tensor*	  v	  = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, qkv, C, N, qkv->nb[1], ( size_t )2 * C * es ) ), hd, H, N );
		q					  = rope( qk_norm( q, blk + ".self_attn.q_rms_norm.gamma" ) );
		k					  = rope( qk_norm( k, blk + ".self_attn.k_rms_norm.gamma" ) );
		ggml_tensor* sa		  = lin( sdpa( q, k, v ), blk + ".self_attn.to_out" );
		h					  = ggml_add( ctx, h, ggml_mul( ctx, sa, gate_msa ) );

		// cross-attention (norm2 affine; QK-RMSNorm, no RoPE, no gate)
		ggml_tensor* h2 = ggml_norm( ctx, h, 1e-6f );
		h2				= ggml_add( ctx, ggml_mul( ctx, h2, W( blk + ".norm2.weight" ) ), W( blk + ".norm2.bias" ) );
		ggml_tensor* cq = ggml_reshape_3d( ctx, lin( h2, blk + ".cross_attn.to_q" ), hd, H, N );
		cq				= qk_norm( cq, blk + ".cross_attn.q_rms_norm.gamma" );
		ggml_tensor* kv = lin( cond_h, blk + ".cross_attn.to_kv" ); // [2C, Lkv]
		ggml_tensor* ck = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, kv, C, Lkv, kv->nb[1], 0 ) ), hd, H, Lkv );
		ggml_tensor* cv = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, kv, C, Lkv, kv->nb[1], ( size_t )C * es ) ), hd, H, Lkv );
		ck				= qk_norm( ck, blk + ".cross_attn.k_rms_norm.gamma" );
		ggml_tensor* ca = lin( sdpa( cq, ck, cv ), blk + ".cross_attn.to_out" );
		h				= ggml_add( ctx, h, ca );

		// feed-forward (norm3 affine-free, modulated; GELU-tanh)
		ggml_tensor* hm = modulate( ggml_norm( ctx, h, 1e-6f ), scale_mlp, shift_mlp );
		hm				= lin( hm, blk + ".mlp.mlp.0" );
		hm				= ggml_gelu( ctx, hm );
		hm				= lin( hm, blk + ".mlp.mlp.2" );
		h				= ggml_add( ctx, h, ggml_mul( ctx, hm, gate_mlp ) );
	}

	// ── head: affine-free LayerNorm (eps 1e-5) + output projection ────────────
	h			   = ggml_norm( ctx, h, 1e-5f );
	h			   = lin( h, "out_layer" );						 // [out_channels, N]
	ggml_tensor* y = ggml_cont( ctx, ggml_transpose( ctx, h ) ); // [N, out_channels], channel-major
	ggml_set_output( y );

	if( !missing.empty() ) {
		set_error( error, "missing tensor: " + missing );
		ggml_free( ctx );
		return false;
	}

	ggml_build_forward_expand( gf, y );
	auto		   t_build = t_now();

	// ── allocate + run on the model's backend (GPU if available, else CPU) ────
	ggml_backend_t backend = m->backend;
	ggml_gallocr_t alloc   = ggml_gallocr_new( ggml_backend_get_default_buffer_type( backend ) );
	if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
		set_error( error, "ggml_gallocr_alloc_graph failed" );
		ggml_gallocr_free( alloc );
		ggml_free( ctx );
		return false;
	}
	auto			   t_alloc = t_now();

	std::vector<float> emb = timestep_embedding( t, 256 );
	std::vector<float> cosv, sinv;
	rope_tables( R, hd, hp.rope_freq_min, hp.rope_freq_base, cosv, sinv );

	ggml_backend_tensor_set( x_t, x, 0, ( size_t )hp.in_channels * N * es );
	ggml_backend_tensor_set( temb, emb.data(), 0, emb.size() * es );
	ggml_backend_tensor_set( cos_t, cosv.data(), 0, cosv.size() * es );
	ggml_backend_tensor_set( sin_t, sinv.data(), 0, sinv.size() * es );
	ggml_backend_tensor_set( cnd, cond, 0, ( size_t )cond_channels * Lkv * es );
	auto			  t_upload = t_now();

	const ggml_status st		= ggml_backend_graph_compute( backend, gf );
	auto			  t_compute = t_now();
	bool			  ok		= ( st == GGML_STATUS_SUCCESS );
	if( ok ) {
		ggml_backend_tensor_get( y, out, 0, ( size_t )hp.out_channels * N * es );
		if( t2_timing ) {
			std::fprintf( stderr,
				"[ss_flow] build=%.1f alloc=%.1f upload=%.1f compute=%.1f read=%.1f total=%.1f ms\n",
				t_ms( t_start, t_build ),
				t_ms( t_build, t_alloc ),
				t_ms( t_alloc, t_upload ),
				t_ms( t_upload, t_compute ),
				t_ms( t_compute, t_now() ),
				t_ms( t_start, t_now() ) );
		}
	} else {
		set_error( error, "graph compute failed" );
	}

	ggml_gallocr_free( alloc );
	ggml_free( ctx );
	return ok;
}

/*****************************************************************************
** Flow-Euler sampler (classifier-free guidance + interval + rescale)
**
** Mirrors FlowEulerGuidanceIntervalSampler.sample. All the per-step flow
** arithmetic is elementwise on the latent and runs on the host; only the
** velocity prediction (1 or 2 forwards per step) uses the GPU graph above.
*****************************************************************************/

namespace
{

// x_0 estimate from a velocity prediction: (1-s)x_t - (s + (1-s)t) pred.
inline void pred_to_xstart( const std::vector<float>& x_t, double t, double sm, const std::vector<float>& pred, std::vector<float>& x0 )
{
	const double a = 1.0 - sm;
	const double b = sm + a * t;
	for( size_t i = 0; i < x_t.size(); ++i )
		x0[i] = ( float )( a * x_t[i] - b * pred[i] );
}

// inverse of pred_to_xstart.
inline void xstart_to_pred( const std::vector<float>& x_t, double t, double sm, const std::vector<float>& x0, std::vector<float>& pred )
{
	const double a = 1.0 - sm;
	const double b = sm + a * t;
	for( size_t i = 0; i < x_t.size(); ++i )
		pred[i] = ( float )( ( a * x_t[i] - x0[i] ) / b );
}

// unbiased std over a whole buffer (matches torch .std(), correction=1).
double unbiased_std( const std::vector<float>& v )
{
	const size_t n = v.size();
	if( n < 2 )
		return 0.0;
	double sum = 0.0;
	for( float x : v )
		sum += x;
	const double mean = sum / ( double )n;
	double		 ss	  = 0.0;
	for( float x : v ) {
		const double d = ( double )x - mean;
		ss += d * d;
	}
	return std::sqrt( ss / ( double )( n - 1 ) );
}

} // namespace

bool trellis2_ss_flow_sample(
	trellis2_ss_flow_model* m, const float* cond, int cond_tokens, int cond_channels, const trellis2_ss_sampler_params* params_in, const float* noise, float* out_latent, std::string* error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}

	trellis2_ss_sampler_params P;
	if( params_in )
		P = *params_in;

	const trellis2_ss_flow_hparams& hp = m->hp;
	const int						R  = hp.resolution;
	const size_t					N  = ( size_t )R * R * R;
	const size_t					n  = ( size_t )hp.in_channels * N;
	const double					sm = P.sigma_min;

	// ── initial noise ─────────────────────────────────────────────────────────
	std::vector<float>				x_t( n );
	if( noise ) {
		std::memcpy( x_t.data(), noise, n * sizeof( float ) );
	} else {
		std::mt19937_64					rng( P.seed );
		std::normal_distribution<float> nd( 0.0f, 1.0f );
		for( size_t i = 0; i < n; ++i )
			x_t[i] = nd( rng );
	}

	// ── timestep schedule: linspace(1,0,steps+1) warped by rescale_t ─────────
	std::vector<double> ts( ( size_t )P.steps + 1 );
	for( int i = 0; i <= P.steps; ++i ) {
		const double lin = 1.0 - ( double )i / ( double )P.steps; // 1 -> 0
		ts[i]			 = P.rescale_t * lin / ( 1.0 + ( P.rescale_t - 1.0 ) * lin );
	}

	const std::vector<float> zero_cond( ( size_t )cond_tokens * cond_channels, 0.0f );
	std::vector<float>		 pred_pos( n ), pred_neg( n ), pred_v( n ), x0_pos( n ), x0_cfg( n );
	std::vector<float>		 x0_view; // scratch for the live-preview x_0 estimate
	if( P.preview )
		x0_view.resize( n );

	auto fwd = [&]( double t, const float* c, std::vector<float>& dst ) -> bool {
		return trellis2_ss_flow_forward( m, x_t.data(), ( float )( 1000.0 * t ), c, cond_tokens, cond_channels, dst.data(), error );
	};

	for( int i = 0; i < P.steps; ++i ) {
		const double t = ts[i], t_prev = ts[i + 1];
		const bool	 in_interval = ( t >= P.guidance_interval_min && t <= P.guidance_interval_max );
		const float	 gs			 = in_interval ? P.guidance_strength : 1.0f;

		if( gs == 1.0f ) {
			if( !fwd( t, cond, pred_v ) )
				return false;
		} else if( gs == 0.0f ) {
			if( !fwd( t, zero_cond.data(), pred_v ) )
				return false;
		} else {
			if( !fwd( t, cond, pred_pos ) )
				return false;
			if( !fwd( t, zero_cond.data(), pred_neg ) )
				return false;
			for( size_t k = 0; k < n; ++k )
				pred_v[k] = gs * pred_pos[k] + ( 1.0f - gs ) * pred_neg[k];

			if( P.guidance_rescale > 0.0f ) {
				pred_to_xstart( x_t, t, sm, pred_pos, x0_pos );
				pred_to_xstart( x_t, t, sm, pred_v, x0_cfg );
				const double std_pos = unbiased_std( x0_pos );
				const double std_cfg = unbiased_std( x0_cfg );
				const double ratio	 = ( std_cfg != 0.0 ) ? std_pos / std_cfg : 1.0;
				const float	 gr		 = P.guidance_rescale;
				for( size_t k = 0; k < n; ++k ) {
					const double rescaled = x0_cfg[k] * ratio;
					x0_cfg[k]			  = ( float )( gr * rescaled + ( 1.0 - gr ) * x0_cfg[k] );
				}
				xstart_to_pred( x_t, t, sm, x0_cfg, pred_v );
			}
		}

		// Live preview: the denoised x_0 estimate at this step (best guess of the
		// clean latent), computed from x_t@t before the Euler step overwrites it.
		if( P.preview ) {
			pred_to_xstart( x_t, t, sm, pred_v, x0_view );
			P.preview( P.preview_user, i + 1, P.steps, x0_view.data(), ( int )n );
		}

		// Euler step: x_{t-1} = x_t - (t - t_prev) * v
		const double dt = t - t_prev;
		for( size_t k = 0; k < n; ++k )
			x_t[k] = ( float )( x_t[k] - dt * pred_v[k] );

		if( P.verbose ) {
			std::fprintf( stderr, "\r[ss sample] step %2d/%d  t=%.4f->%.4f  %s   ", i + 1, P.steps, t, t_prev, in_interval ? "cfg" : "uncond" );
			std::fflush( stderr );
		}
		if( P.progress )
			P.progress( P.progress_user, i + 1, P.steps );
	}
	if( P.verbose )
		std::fprintf( stderr, "\n" );

	std::memcpy( out_latent, x_t.data(), n * sizeof( float ) );
	return true;
}

/*****************************************************************************
** Sparse-structure decoder (stage 1): SparseStructureDecoder
**
**   h = input_layer(z_s)                         # Conv3d latent->channels[0]
**   h = middle_block(h)                          # num_res_blocks_middle ResBlocks
**   for level i in 0..n_levels-1:
**       h = ResBlock x num_res_blocks            # at channels[i]
**       if i < n_levels-1: h = Upsample(h)       # Conv3d (C->C'*8) + pixel_shuffle_3d
**   logits = out_layer(h)                        # ChannelLayerNorm + SiLU + Conv3d->out
**
** ResBlock3d: x + conv2(silu(norm2(conv1(silu(norm1(x)))))), all skips Identity
** here (in==out at every block). norm is a per-voxel LayerNorm over channels.
** Two pixel-shuffle upsamples take 16^3 -> 32^3 -> 64^3.
*****************************************************************************/

struct trellis2_ss_dec_model {
	gguf_context*								  gguf = nullptr;
	ggml_context*								  ctx  = nullptr;
	trellis2_ss_dec_hparams						  hp;
	bool										  has_data = false;

	ggml_backend_t								  backend	  = nullptr;
	ggml_backend_buffer_t						  weights_buf = nullptr;
	std::string									  backend_name;

	std::unordered_map<std::string, ggml_tensor*> tensors;
};

trellis2_ss_dec_model* trellis2_ss_dec_load( const std::string& path, bool load_tensors, std::string* error, const char* device )
{
	auto*			 m = new trellis2_ss_dec_model();

	gguf_init_params params;
	params.no_alloc = true;
	params.ctx		= &m->ctx;

	m->gguf = gguf_init_from_file( path.c_str(), params );
	if( !m->gguf ) {
		set_error( error, "gguf_init_from_file failed (not a GGUF file?): " + path );
		delete m;
		return nullptr;
	}

	const char* arch = kv_str( m->gguf, "general.architecture", "" );
	if( std::strcmp( arch, "trellis2-ss-dec" ) != 0 ) {
		set_error( error, std::string( "unexpected architecture '" ) + arch + "' (expected 'trellis2-ss-dec')" );
		trellis2_ss_dec_free( m );
		return nullptr;
	}

	trellis2_ss_dec_hparams& hp = m->hp;
	const char*				 P	= "trellis2.ss_dec.";
	auto					 K	= [&]( const char* suffix ) { return std::string( P ) + suffix; };

	hp.out_channels			 = ( int32_t )kv_u32( m->gguf, K( "out_channels" ).c_str(), 1 );
	hp.latent_channels		 = ( int32_t )kv_u32( m->gguf, K( "latent_channels" ).c_str(), 8 );
	hp.num_res_blocks		 = ( int32_t )kv_u32( m->gguf, K( "num_res_blocks" ).c_str(), 2 );
	hp.num_res_blocks_middle = ( int32_t )kv_u32( m->gguf, K( "num_res_blocks_middle" ).c_str(), 2 );
	hp.n_levels				 = ( int32_t )kv_u32( m->gguf, K( "n_levels" ).c_str(), 3 );
	hp.norm_eps				 = kv_f32( m->gguf, K( "norm_eps" ).c_str(), 1e-5f );
	hp.file_type			 = ( int32_t )kv_u32( m->gguf, "general.file_type", 0 );
	std::snprintf( hp.norm_type, sizeof( hp.norm_type ), "%s", kv_str( m->gguf, K( "norm_type" ).c_str(), "layer" ) );
	if( hp.n_levels > 8 )
		hp.n_levels = 8;
	for( int i = 0; i < hp.n_levels; ++i ) {
		hp.channels[i] = ( int32_t )kv_u32( m->gguf, K( ( "channels." + std::to_string( i ) ).c_str() ).c_str(), 0 );
	}

	for( ggml_tensor* t = ggml_get_first_tensor( m->ctx ); t != nullptr; t = ggml_get_next_tensor( m->ctx, t ) ) {
		m->tensors[t->name] = t;
	}

	if( load_tensors ) {
		m->backend	   = init_best_backend( m->backend_name, device );
		m->weights_buf = ggml_backend_alloc_ctx_tensors( m->ctx, m->backend );
		if( !m->weights_buf ) {
			set_error( error, "failed to allocate weights on backend " + m->backend_name );
			trellis2_ss_dec_free( m );
			return nullptr;
		}
		std::ifstream fin( path, std::ios::binary );
		if( !fin ) {
			set_error( error, "cannot reopen file for weight data: " + path );
			trellis2_ss_dec_free( m );
			return nullptr;
		}
		const size_t		 data_off = gguf_get_data_offset( m->gguf );
		const int64_t		 nt		  = gguf_get_n_tensors( m->gguf );
		std::vector<uint8_t> buf;
		for( int64_t i = 0; i < nt; ++i ) {
			const char*	 name = gguf_get_tensor_name( m->gguf, i );
			ggml_tensor* t	  = m->tensors[name];
			const size_t nb	  = ggml_nbytes( t );
			const size_t off  = data_off + gguf_get_tensor_offset( m->gguf, i );
			buf.resize( nb );
			fin.seekg( ( std::streamoff )off, std::ios::beg );
			if( !fin.read( reinterpret_cast<char*>( buf.data() ), ( std::streamsize )nb ) ) {
				set_error( error, std::string( "failed reading weight '" ) + name + "' from file" );
				trellis2_ss_dec_free( m );
				return nullptr;
			}
			ggml_backend_tensor_set( t, buf.data(), 0, nb );
		}
		m->has_data = true;
	}

	return m;
}

void trellis2_ss_dec_free( trellis2_ss_dec_model* m )
{
	if( !m )
		return;
	if( m->weights_buf )
		ggml_backend_buffer_free( m->weights_buf );
	if( m->backend )
		ggml_backend_free( m->backend );
	if( m->gguf )
		gguf_free( m->gguf );
	if( m->ctx )
		ggml_free( m->ctx );
	delete m;
}

const char* trellis2_ss_dec_backend_name( const trellis2_ss_dec_model* m )
{
	return ( m && !m->backend_name.empty() ) ? m->backend_name.c_str() : "none";
}

const trellis2_ss_dec_hparams& trellis2_ss_dec_hparams_of( const trellis2_ss_dec_model* m )
{
	return m->hp;
}

int trellis2_ss_dec_n_tensors( const trellis2_ss_dec_model* m )
{
	return m ? ( int )gguf_get_n_tensors( m->gguf ) : 0;
}

bool trellis2_ss_dec_get_tensor_info( const trellis2_ss_dec_model* m, int i, trellis2_tensor_info& out )
{
	if( !m || i < 0 || i >= ( int )gguf_get_n_tensors( m->gguf ) )
		return false;
	const char* name = gguf_get_tensor_name( m->gguf, i );
	out.name		 = name;
	ggml_tensor* t	 = ggml_get_tensor( m->ctx, name );
	if( !t )
		return false;
	out.n_dims = ggml_n_dims( t );
	for( int d = 0; d < 4; ++d )
		out.ne[d] = t->ne[d];
	out.ggml_type = ( int )t->type;
	out.type_name = ggml_type_name( t->type );
	out.n_bytes	  = ggml_nbytes( t );
	return true;
}

bool trellis2_ss_dec_decode( trellis2_ss_dec_model* m, const float* latent, float* out, std::string* error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}

	const trellis2_ss_dec_hparams& hp  = m->hp;
	const int					   R   = hp.res_in();		 // 16
	const int					   Cin = hp.latent_channels; // 8
	const float					   eps = hp.norm_eps;		 // 1e-5
	const size_t				   es  = sizeof( float );

	std::string					   missing;
	auto						   W = [&]( const std::string& n ) -> ggml_tensor* {
		  auto it = m->tensors.find( n );
		  if( it == m->tensors.end() ) {
			  if( missing.empty() )
				  missing = n;
			  return nullptr;
		  }
		  return it->second;
	};

	const size_t	 mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom( 8192, false );
	ggml_init_params ip { mem, nullptr, /*no_alloc*/ true };
	ggml_context*	 ctx = ggml_init( ip );
	ggml_cgraph*	 gf	 = ggml_new_graph_custom( ctx, 8192, false );

	// input leaf: z_s as ggml [R,R,R,Cin] (ne0=k, ne1=j, ne2=i, ne3=channel) —
	// identical layout to channel-major latent[c*R^3 + i*R^2 + j*R + k].
	ggml_tensor*	 x = ggml_new_tensor_4d( ctx, GGML_TYPE_F32, R, R, R, Cin );
	ggml_set_input( x );
	ggml_set_name( x, "z_s" );

	// Conv3d (stride 1, pad 1) + per-output-channel bias. ic/oc passed explicitly.
	auto conv = [&]( ggml_tensor* in, const std::string& pfx, int ic, int oc ) -> ggml_tensor* {
		ggml_tensor* w = W( pfx + ".weight" );
		ggml_tensor* b = W( pfx + ".bias" );
		if( !w )
			return in;
		ggml_tensor* y = ggml_conv_3d_direct( ctx, w, in, 1, 1, 1, 1, 1, 1, 1, 1, 1, ic, 1, oc );
		if( b )
			y = ggml_add( ctx, y, ggml_reshape_4d( ctx, b, 1, 1, 1, oc ) );
		return y;
	};
	// ChannelLayerNorm32: per-voxel LayerNorm over the channel axis (with affine).
	auto clnorm = [&]( ggml_tensor* in, const std::string& pfx ) -> ggml_tensor* {
		ggml_tensor* p = ggml_cont( ctx, ggml_permute( ctx, in, 1, 2, 3, 0 ) ); // [C,W,H,D]
		p			   = ggml_norm( ctx, p, eps );
		p			   = ggml_mul( ctx, p, W( pfx + ".weight" ) );
		p			   = ggml_add( ctx, p, W( pfx + ".bias" ) );
		return ggml_cont( ctx, ggml_permute( ctx, p, 3, 0, 1, 2 ) ); // [W,H,D,C]
	};
	auto resblock = [&]( ggml_tensor* in, const std::string& pfx, int C ) -> ggml_tensor* {
		ggml_tensor* h = clnorm( in, pfx + ".norm1" );
		h			   = ggml_silu( ctx, h );
		h			   = conv( h, pfx + ".conv1", C, C );
		h			   = clnorm( h, pfx + ".norm2" );
		h			   = ggml_silu( ctx, h );
		h			   = conv( h, pfx + ".conv2", C, C );
		return ggml_add( ctx, h, in ); // skip = Identity
	};
	// pixel_shuffle_3d(scale 2): [A,A,A, Cout*8] -> [2A,2A,2A, Cout]. Each scale
	// bit (LSB->axis0, mid->axis1, MSB->axis2, matching torch's H/W/D pairing)
	// is peeled out of the channel and interleaved into its spatial axis.
	auto pshuf = [&]( ggml_tensor* t, int A0, int A1, int A2, int Co ) -> ggml_tensor* {
		// peel s2 (channel LSB) into axis0
		t = ggml_reshape_4d( ctx, t, A0, A1 * A2, 2, Co * 4 );
		t = ggml_cont( ctx, ggml_permute( ctx, t, 1, 2, 0, 3 ) ); // [2, A0, A1*A2, Co*4]
		t = ggml_reshape_4d( ctx, t, 2 * A0, A1, A2, Co * 4 );
		// peel s1 into axis1
		t = ggml_cont( ctx, ggml_permute( ctx, t, 1, 0, 2, 3 ) ); // [A1, 2A0, A2, Co*4]
		t = ggml_reshape_4d( ctx, t, A1, 2 * A0 * A2, 2, Co * 2 );
		t = ggml_cont( ctx, ggml_permute( ctx, t, 1, 2, 0, 3 ) ); // [2, A1, 2A0*A2, Co*2]
		t = ggml_reshape_4d( ctx, t, 2 * A1, 2 * A0, A2, Co * 2 );
		t = ggml_cont( ctx, ggml_permute( ctx, t, 1, 0, 2, 3 ) ); // [2A0, 2A1, A2, Co*2]
		// peel s0 (channel MSB) into axis2
		t = ggml_cont( ctx, ggml_permute( ctx, t, 1, 2, 0, 3 ) ); // [A2, 2A0, 2A1, Co*2]
		t = ggml_reshape_4d( ctx, t, A2, 2 * A0 * 2 * A1, 2, Co );
		t = ggml_cont( ctx, ggml_permute( ctx, t, 1, 2, 0, 3 ) ); // [2, A2, 2A0*2A1, Co]
		t = ggml_reshape_4d( ctx, t, 2 * A2, 2 * A0, 2 * A1, Co );
		t = ggml_cont( ctx, ggml_permute( ctx, t, 2, 0, 1, 3 ) ); // [2A0, 2A1, 2A2, Co]
		return t;
	};

	// ── forward ───────────────────────────────────────────────────────────────
	ggml_tensor* h = conv( x, "input_layer", Cin, hp.channels[0] );

	for( int i = 0; i < hp.num_res_blocks_middle; ++i ) {
		h = resblock( h, "middle_block." + std::to_string( i ), hp.channels[0] );
	}

	int blk		= 0;
	int cur_res = R;
	for( int lvl = 0; lvl < hp.n_levels; ++lvl ) {
		const int C = hp.channels[lvl];
		for( int r = 0; r < hp.num_res_blocks; ++r ) {
			h = resblock( h, "blocks." + std::to_string( blk++ ), C );
		}
		if( lvl < hp.n_levels - 1 ) {
			const int Co = hp.channels[lvl + 1];
			h			 = conv( h, "blocks." + std::to_string( blk++ ) + ".conv", C, Co * 8 );
			h			 = pshuf( h, cur_res, cur_res, cur_res, Co );
			cur_res *= 2;
		}
	}

	h = clnorm( h, "out_layer.0" );
	h = ggml_silu( ctx, h );
	h = conv( h, "out_layer.2", hp.channels[hp.n_levels - 1], hp.out_channels ); // [Rout,Rout,Rout,Oc]
	ggml_set_output( h );

	if( !missing.empty() ) {
		set_error( error, "missing tensor: " + missing );
		ggml_free( ctx );
		return false;
	}

	ggml_build_forward_expand( gf, h );

	ggml_backend_t backend = m->backend;
	ggml_gallocr_t alloc   = ggml_gallocr_new( ggml_backend_get_default_buffer_type( backend ) );
	if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
		set_error( error, "ggml_gallocr_alloc_graph failed" );
		ggml_gallocr_free( alloc );
		ggml_free( ctx );
		return false;
	}

	ggml_backend_tensor_set( x, latent, 0, ( size_t )Cin * R * R * R * es );

	const ggml_status st = ggml_backend_graph_compute( backend, gf );
	bool			  ok = ( st == GGML_STATUS_SUCCESS );
	if( ok ) {
		const size_t Rout = ( size_t )hp.res_out();
		ggml_backend_tensor_get( h, out, 0, ( size_t )hp.out_channels * Rout * Rout * Rout * es );
	} else {
		set_error( error, "graph compute failed" );
	}

	ggml_gallocr_free( alloc );
	ggml_free( ctx );
	return ok;
}

/*****************************************************************************
** DINOv3 ViT-L/16 image-conditioning encoder — GGUF loader
*****************************************************************************/

struct trellis2_dino_model {
	gguf_context*								  gguf = nullptr;
	ggml_context*								  ctx  = nullptr;
	trellis2_dino_hparams						  hp;
	bool										  has_data = false;

	ggml_backend_t								  backend	  = nullptr;
	ggml_backend_buffer_t						  weights_buf = nullptr;
	std::string									  backend_name;

	std::unordered_map<std::string, ggml_tensor*> tensors;
};

trellis2_dino_model* trellis2_dino_load( const std::string& path, bool load_tensors, std::string* error, const char* device )
{
	auto*			 m = new trellis2_dino_model();

	gguf_init_params params;
	params.no_alloc = true;
	params.ctx		= &m->ctx;

	m->gguf = gguf_init_from_file( path.c_str(), params );
	if( !m->gguf ) {
		set_error( error, "gguf_init_from_file failed (not a GGUF file?): " + path );
		delete m;
		return nullptr;
	}

	const char* arch = kv_str( m->gguf, "general.architecture", "" );
	if( std::strcmp( arch, "trellis2-dino" ) != 0 ) {
		set_error( error, std::string( "unexpected architecture '" ) + arch + "' (expected 'trellis2-dino')" );
		trellis2_dino_free( m );
		return nullptr;
	}

	trellis2_dino_hparams& hp = m->hp;
	const char*			   P  = "trellis2.dino.";
	auto				   K  = [&]( const char* suffix ) { return std::string( P ) + suffix; };

	hp.hidden_size		   = ( int32_t )kv_u32( m->gguf, K( "hidden_size" ).c_str(), 0 );
	hp.n_layers			   = ( int32_t )kv_u32( m->gguf, K( "n_layers" ).c_str(), 0 );
	hp.n_heads			   = ( int32_t )kv_u32( m->gguf, K( "n_heads" ).c_str(), 0 );
	hp.intermediate_size   = ( int32_t )kv_u32( m->gguf, K( "intermediate_size" ).c_str(), 0 );
	hp.patch_size		   = ( int32_t )kv_u32( m->gguf, K( "patch_size" ).c_str(), 16 );
	hp.num_register_tokens = ( int32_t )kv_u32( m->gguf, K( "num_register_tokens" ).c_str(), 0 );
	hp.layer_norm_eps	   = kv_f32( m->gguf, K( "layer_norm_eps" ).c_str(), 1e-5f );
	hp.rope_theta		   = kv_f32( m->gguf, K( "rope_theta" ).c_str(), 100.0f );
	hp.file_type		   = ( int32_t )kv_u32( m->gguf, "general.file_type", 0 );
	for( int c = 0; c < 3; ++c ) {
		hp.image_mean[c] = kv_f32( m->gguf, ( K( "image_mean." ) + std::to_string( c ) ).c_str(), hp.image_mean[c] );
		hp.image_std[c]	 = kv_f32( m->gguf, ( K( "image_std." ) + std::to_string( c ) ).c_str(), hp.image_std[c] );
	}

	for( ggml_tensor* t = ggml_get_first_tensor( m->ctx ); t != nullptr; t = ggml_get_next_tensor( m->ctx, t ) ) {
		m->tensors[t->name] = t;
	}

	if( load_tensors ) {
		m->backend	   = init_best_backend( m->backend_name, device );
		m->weights_buf = ggml_backend_alloc_ctx_tensors( m->ctx, m->backend );
		if( !m->weights_buf ) {
			set_error( error, "failed to allocate weights on backend " + m->backend_name );
			trellis2_dino_free( m );
			return nullptr;
		}

		std::ifstream fin( path, std::ios::binary );
		if( !fin ) {
			set_error( error, "cannot reopen file for weight data: " + path );
			trellis2_dino_free( m );
			return nullptr;
		}
		const size_t		 data_off = gguf_get_data_offset( m->gguf );
		const int64_t		 nt		  = gguf_get_n_tensors( m->gguf );
		std::vector<uint8_t> buf;
		for( int64_t i = 0; i < nt; ++i ) {
			const char*	 name = gguf_get_tensor_name( m->gguf, i );
			ggml_tensor* t	  = m->tensors[name];
			const size_t nb	  = ggml_nbytes( t );
			const size_t off  = data_off + gguf_get_tensor_offset( m->gguf, i );
			buf.resize( nb );
			fin.seekg( ( std::streamoff )off, std::ios::beg );
			if( !fin.read( reinterpret_cast<char*>( buf.data() ), ( std::streamsize )nb ) ) {
				set_error( error, std::string( "failed reading weight '" ) + name + "' from file" );
				trellis2_dino_free( m );
				return nullptr;
			}
			ggml_backend_tensor_set( t, buf.data(), 0, nb );
		}
		m->has_data = true;
	}

	return m;
}

void trellis2_dino_free( trellis2_dino_model* m )
{
	if( !m )
		return;
	if( m->weights_buf )
		ggml_backend_buffer_free( m->weights_buf );
	if( m->backend )
		ggml_backend_free( m->backend );
	if( m->gguf )
		gguf_free( m->gguf );
	if( m->ctx )
		ggml_free( m->ctx );
	delete m;
}

const char* trellis2_dino_backend_name( const trellis2_dino_model* m )
{
	return ( m && !m->backend_name.empty() ) ? m->backend_name.c_str() : "none";
}

const trellis2_dino_hparams& trellis2_dino_hparams_of( const trellis2_dino_model* m )
{
	return m->hp;
}

/*****************************************************************************
** DINOv3 — forward pass
**
** Mirrors DinoV3FeatureExtractor.extract_features:
**   h = embeddings(pixels)            # patch conv + [CLS | 4 reg | patches]
**   cos,sin = rope_embeddings(pixels) # axial 2D RoPE over patch centers
**   for each of 24 layers:            # pre-norm ViT block with LayerScale
**     h += ls1 * attn(LN1(h))         #   (RoPE on patch tokens only)
**     h += ls2 * mlp(LN2(h))          #   (exact-GELU MLP)
**   cond = layer_norm(h)              # affine-free; model.norm NOT applied
*****************************************************************************/

namespace
{

// Axial 2D RoPE tables for a Wp x Hp patch grid, matching HF's
// DINOv3ViTRopePositionEmbedding in eval mode (no shift/jitter/rescale):
//   inv_freq[j] = 1 / theta^(4j/hd),  j < hd/4
//   angles(p)   = 2*pi * [cy, cx] (x) inv_freq   -> [hd/2], tiled to [hd]
// with (cy, cx) the patch-center coords normalized to [-1, 1].
// Output buffers are [P][hd] row-major (== ggml ne [hd, 1, P]).
void dino_rope_tables( int hp_grid, int wp_grid, int head_dim, float theta, std::vector<float>& cos_t, std::vector<float>& sin_t )
{
	const int			quarter = head_dim / 4;
	std::vector<double> inv_freq( ( size_t )quarter );
	for( int j = 0; j < quarter; ++j ) {
		inv_freq[j] = 1.0 / std::pow( ( double )theta, ( double )j * 4.0 / ( double )head_dim );
	}

	const int P = hp_grid * wp_grid;
	cos_t.resize( ( size_t )P * head_dim );
	sin_t.resize( ( size_t )P * head_dim );
	for( int py = 0; py < hp_grid; ++py ) {
		const double cy = 2.0 * ( ( ( double )py + 0.5 ) / ( double )hp_grid ) - 1.0;
		for( int px = 0; px < wp_grid; ++px ) {
			const double cx	  = 2.0 * ( ( ( double )px + 0.5 ) / ( double )wp_grid ) - 1.0;
			const size_t base = ( size_t )( py * wp_grid + px ) * head_dim;
			for( int j = 0; j < quarter; ++j ) {
				const double ay = 2.0 * M_PI * cy * inv_freq[j];
				const double ax = 2.0 * M_PI * cx * inv_freq[j];
				// angles layout: [cy*f..., cx*f...] then tiled x2
				const float	 cy_c = ( float )std::cos( ay ), cy_s = ( float )std::sin( ay );
				const float	 cx_c = ( float )std::cos( ax ), cx_s = ( float )std::sin( ax );
				cos_t[base + j]				  = cy_c;
				cos_t[base + quarter + j]	  = cx_c;
				cos_t[base + 2 * quarter + j] = cy_c;
				cos_t[base + 3 * quarter + j] = cx_c;
				sin_t[base + j]				  = cy_s;
				sin_t[base + quarter + j]	  = cx_s;
				sin_t[base + 2 * quarter + j] = cy_s;
				sin_t[base + 3 * quarter + j] = cx_s;
			}
		}
	}
}

} // namespace

bool trellis2_dino_encode( trellis2_dino_model* m, const float* pixel_values, int image_size, trellis2_dino_cond& out, trellis2_dino_taps* taps, std::string* error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}

	const trellis2_dino_hparams& hp = m->hp;
	const int					 S	= image_size;
	if( S <= 0 || S % hp.patch_size != 0 ) {
		set_error( error, "image_size must be a positive multiple of patch_size" );
		return false;
	}
	const int	Wp		   = S / hp.patch_size;			 // 32 @ 512
	const int	P		   = Wp * Wp;					 // 1024 patch tokens
	const int	Npre	   = 1 + hp.num_register_tokens; // CLS + registers
	const int	N		   = Npre + P;					 // 1029 tokens
	const int	C		   = hp.hidden_size;			 // 1024
	const int	H		   = hp.n_heads;				 // 16
	const int	hd		   = hp.head_dim();				 // 64
	const float attn_scale = 1.0f / std::sqrt( ( float )hd );
	const float eps		   = hp.layer_norm_eps;

	std::string missing;
	auto		W = [&]( const std::string& n ) -> ggml_tensor* {
		   auto it = m->tensors.find( n );
		   if( it == m->tensors.end() ) {
			   if( missing.empty() )
				   missing = n;
			   return nullptr;
		   }
		   return it->second;
	};
	auto Wopt = [&]( const std::string& n ) -> ggml_tensor* {
		auto it = m->tensors.find( n );
		return it == m->tensors.end() ? nullptr : it->second;
	};

	const size_t	 mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom( 8192, false );
	ggml_init_params ip { mem, nullptr, /*no_alloc*/ true };
	ggml_context*	 ctx = ggml_init( ip );
	ggml_cgraph*	 gf	 = ggml_new_graph_custom( ctx, 8192, false );

	// ── input leaves ─────────────────────────────────────────────────────────
	ggml_tensor*	 pix   = ggml_new_tensor_4d( ctx, GGML_TYPE_F32, S, S, 3, 1 ); // CHW flat
	ggml_tensor*	 cos_t = ggml_new_tensor_3d( ctx, GGML_TYPE_F32, hd, 1, P );
	ggml_tensor*	 sin_t = ggml_new_tensor_3d( ctx, GGML_TYPE_F32, hd, 1, P );
	ggml_set_input( pix );
	ggml_set_input( cos_t );
	ggml_set_input( sin_t );

	// Tap bookkeeping: tensors registered here are marked as graph outputs and
	// copied to the host after compute. All taps are [C, N]-contiguous, whose
	// memory order equals the reference's row-major [N, C].
	std::vector<std::pair<std::string, ggml_tensor*>> tap_list;
	auto											  tap = [&]( const std::string& name, ggml_tensor* t ) {
		 if( !taps )
			 return;
		 ggml_tensor* c = ggml_cont( ctx, t );
		 ggml_set_output( c );
		 ggml_build_forward_expand( gf, c );
		 tap_list.emplace_back( name, c );
	};
	tap( "rope_0", cos_t ); // [hd,1,P] contiguous == reference's [P, hd]
	tap( "rope_1", sin_t );

	auto lin = [&]( ggml_tensor* in, const std::string& pfx ) -> ggml_tensor* {
		ggml_tensor* y = ggml_mul_mat( ctx, W( pfx + ".weight" ), in );
		ggml_tensor* b = Wopt( pfx + ".bias" );
		if( b )
			y = ggml_add( ctx, y, b );
		return y;
	};
	// affine LayerNorm over channels of a [C, N] tensor
	auto lnorm = [&]( ggml_tensor* h, const std::string& pfx ) -> ggml_tensor* {
		ggml_tensor* y = ggml_norm( ctx, h, eps );
		y			   = ggml_mul( ctx, y, W( pfx + ".weight" ) );
		y			   = ggml_add( ctx, y, W( pfx + ".bias" ) );
		return y;
	};
	// half-split RoPE on a [hd, H, P] tensor: x*cos + rotate_half(x)*sin
	auto rope = [&]( ggml_tensor* q3 ) -> ggml_tensor* {
		const size_t half_off = ( size_t )( hd / 2 ) * sizeof( float );
		ggml_tensor* x1		  = ggml_cont( ctx, ggml_view_3d( ctx, q3, hd / 2, H, P, q3->nb[1], q3->nb[2], 0 ) );
		ggml_tensor* x2		  = ggml_cont( ctx, ggml_view_3d( ctx, q3, hd / 2, H, P, q3->nb[1], q3->nb[2], half_off ) );
		ggml_tensor* rh		  = ggml_concat( ctx, ggml_neg( ctx, x2 ), x1, 0 ); // [hd,H,P]
		return ggml_add( ctx, ggml_mul( ctx, q3, cos_t ), ggml_mul( ctx, rh, sin_t ) );
	};
	// RoPE on patch tokens only of a [hd, H, N] tensor (prefix passes through).
	auto rope_patches = [&]( ggml_tensor* q3 ) -> ggml_tensor* {
		ggml_tensor* pre = ggml_cont( ctx, ggml_view_3d( ctx, q3, hd, H, Npre, q3->nb[1], q3->nb[2], 0 ) );
		ggml_tensor* pat = ggml_cont( ctx, ggml_view_3d( ctx, q3, hd, H, P, q3->nb[1], q3->nb[2], ( size_t )Npre * q3->nb[2] ) );
		return ggml_concat( ctx, pre, rope( pat ), 2 );
	};
	// scaled-dot-product attention; q3/k3/v3 are [hd, H, N].
	auto sdpa = [&]( ggml_tensor* q3, ggml_tensor* k3, ggml_tensor* v3 ) -> ggml_tensor* {
		ggml_tensor* qp = ggml_cont( ctx, ggml_permute( ctx, q3, 0, 2, 1, 3 ) ); // [hd, N, H]
		ggml_tensor* kp = ggml_cont( ctx, ggml_permute( ctx, k3, 0, 2, 1, 3 ) );
		ggml_tensor* vp = ggml_cont( ctx, ggml_permute( ctx, v3, 0, 2, 1, 3 ) );
		ggml_tensor* sc = ggml_mul_mat( ctx, kp, qp ); // [Nk, Nq, H]
		sc				= ggml_soft_max_ext( ctx, sc, nullptr, attn_scale, 0.0f );
		ggml_tensor* vt = ggml_cont( ctx, ggml_permute( ctx, vp, 1, 0, 2, 3 ) ); // [Nk, hd, H]
		ggml_tensor* o	= ggml_mul_mat( ctx, vt, sc );							 // [hd, Nq, H]
		o				= ggml_cont( ctx, ggml_permute( ctx, o, 0, 2, 1, 3 ) );	 // [hd, H, Nq]
		return ggml_reshape_2d( ctx, o, C, o->ne[2] );							 // [C, Nq]
	};

	// ── embeddings: patch conv + CLS + register tokens ───────────────────────
	ggml_tensor* pconv = ggml_conv_2d( ctx, W( "embeddings.patch_embeddings.weight" ), pix, hp.patch_size, hp.patch_size, 0, 0, 1, 1 ); // [Wp, Wp, C, 1]
	pconv			   = ggml_reshape_2d( ctx, pconv, P, C );																			// [P, C] (token-fastest)
	pconv			   = ggml_cont( ctx, ggml_transpose( ctx, pconv ) );																// [C, P]
	pconv			   = ggml_add( ctx, pconv, W( "embeddings.patch_embeddings.bias" ) );

	ggml_tensor* h = ggml_concat( ctx, W( "embeddings.cls_token" ), ggml_concat( ctx, W( "embeddings.register_tokens" ), pconv, 1 ), 1 ); // [C, N]
	h			   = ggml_cont( ctx, h );
	tap( "embd", h );

	// ── transformer layers ───────────────────────────────────────────────────
	const bool detail_first_last = true;
	for( int i = 0; i < hp.n_layers; ++i ) {
		const std::string blk	 = "layer." + std::to_string( i );
		const bool		  detail = taps && detail_first_last && ( i == 0 || i == hp.n_layers - 1 );
		auto			  tn	 = [&]( const char* s ) { return "l" + std::to_string( i ) + "." + s; };
		( void )tn;

		ggml_tensor* hn = lnorm( h, blk + ".norm1" );
		if( detail )
			tap( tn( "norm1" ), hn );

		ggml_tensor* q	= ggml_reshape_3d( ctx, lin( hn, blk + ".attention.q_proj" ), hd, H, N );
		ggml_tensor* k	= ggml_reshape_3d( ctx, lin( hn, blk + ".attention.k_proj" ), hd, H, N );
		ggml_tensor* v	= ggml_reshape_3d( ctx, lin( hn, blk + ".attention.v_proj" ), hd, H, N );
		q				= rope_patches( q );
		k				= rope_patches( k );
		ggml_tensor* sa = lin( sdpa( q, k, v ), blk + ".attention.o_proj" ); // [C, N]
		if( detail )
			tap( tn( "attention" ), sa );

		ggml_tensor* ls1 = ggml_mul( ctx, sa, W( blk + ".layer_scale1.lambda1" ) );
		if( detail )
			tap( tn( "layer_scale1" ), ls1 );
		h = ggml_add( ctx, h, ls1 );

		ggml_tensor* h2 = lnorm( h, blk + ".norm2" );
		if( detail )
			tap( tn( "norm2" ), h2 );
		ggml_tensor* mlp = lin( h2, blk + ".mlp.up_proj" );
		mlp				 = ggml_gelu_erf( ctx, mlp );
		mlp				 = lin( mlp, blk + ".mlp.down_proj" );
		if( detail )
			tap( tn( "mlp" ), mlp );
		ggml_tensor* ls2 = ggml_mul( ctx, mlp, W( blk + ".layer_scale2.lambda1" ) );
		if( detail )
			tap( tn( "layer_scale2" ), ls2 );
		h = ggml_add( ctx, h, ls2 );

		tap( "l" + std::to_string( i ) + ".out", h );
	}

	// ── affine-free final LayerNorm (F.layer_norm, eps 1e-5) ─────────────────
	ggml_tensor* cond = ggml_norm( ctx, h, 1e-5f );
	cond			  = ggml_cont( ctx, cond ); // [C, N] contiguous == row-major [N, C]
	ggml_set_output( cond );
	tap( "cond", cond );

	if( !missing.empty() ) {
		set_error( error, "missing tensor: " + missing );
		ggml_free( ctx );
		return false;
	}

	ggml_build_forward_expand( gf, cond );

	ggml_backend_t backend = m->backend;
	ggml_gallocr_t alloc   = ggml_gallocr_new( ggml_backend_get_default_buffer_type( backend ) );
	if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
		set_error( error, "ggml_gallocr_alloc_graph failed" );
		ggml_gallocr_free( alloc );
		ggml_free( ctx );
		return false;
	}

	std::vector<float> cosv, sinv;
	dino_rope_tables( Wp, Wp, hd, hp.rope_theta, cosv, sinv );

	const size_t es = sizeof( float );
	ggml_backend_tensor_set( pix, pixel_values, 0, ( size_t )3 * S * S * es );
	ggml_backend_tensor_set( cos_t, cosv.data(), 0, cosv.size() * es );
	ggml_backend_tensor_set( sin_t, sinv.data(), 0, sinv.size() * es );

	const ggml_status st = ggml_backend_graph_compute( backend, gf );
	bool			  ok = ( st == GGML_STATUS_SUCCESS );
	if( ok ) {
		out.shape = { 1, ( int64_t )N, ( int64_t )C };
		out.data.resize( ( size_t )N * C );
		out.format_version = 1;
		ggml_backend_tensor_get( cond, out.data.data(), 0, out.data.size() * es );

		if( taps ) {
			for( auto& nt : tap_list ) {
				taps->names.push_back( nt.first );
				std::vector<float> buf( ggml_nelements( nt.second ) );
				ggml_backend_tensor_get( nt.second, buf.data(), 0, buf.size() * es );
				taps->data.push_back( std::move( buf ) );
			}
		}
	} else {
		set_error( error, "graph compute failed" );
	}

	ggml_gallocr_free( alloc );
	ggml_free( ctx );
	return ok;
}

bool trellis2_dino_encode_rgb( trellis2_dino_model* m, const uint8_t* rgb, int image_size, trellis2_dino_cond& out, std::string* error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	const trellis2_dino_hparams& hp = m->hp;
	const size_t				 S	= ( size_t )image_size;
	std::vector<float>			 pix( 3 * S * S );
	for( int c = 0; c < 3; ++c ) {
		const float mean = hp.image_mean[c], sd = hp.image_std[c];
		for( size_t i = 0; i < S * S; ++i ) {
			pix[( size_t )c * S * S + i] = ( ( float )rgb[i * 3 + c] / 255.0f - mean ) / sd;
		}
	}
	return trellis2_dino_encode( m, pix.data(), image_size, out, nullptr, error );
}

/*****************************************************************************
** Image preprocessing (pipeline.preprocess_image, has_alpha path)
**
** The resampler reproduces PIL's 8-bit fixed-point separable Lanczos-3
** (Pillow Resample.c): double-precision coefficient windows normalized per
** output pixel, quantized to integers at PRECISION_BITS, horizontal pass then
** vertical pass with uint8 rounding between passes. This makes the C++
** preprocessing byte-identical to the Python reference on the same input.
*****************************************************************************/

namespace
{

constexpr int PIL_PRECISION_BITS = 32 - 8 - 2;

inline double pil_sinc( double x )
{
	if( x == 0.0 )
		return 1.0;
	const double px = M_PI * x;
	return std::sin( px ) / px;
}
inline double pil_lanczos3( double x )
{
	if( x >= -3.0 && x < 3.0 )
		return pil_sinc( x ) * pil_sinc( x / 3.0 );
	return 0.0;
}

inline uint8_t pil_clip8( int64_t in )
{
	if( in >= ( ( int64_t )1 << PIL_PRECISION_BITS << 8 ) )
		return 255;
	if( in <= 0 )
		return 0;
	return ( uint8_t )( in >> PIL_PRECISION_BITS );
}

// Coefficient windows for one axis (PIL precompute_coeffs + normalize_8bpc).
void pil_coeffs( int in_size, int out_size, std::vector<int>& bounds, std::vector<int32_t>& kk, int& ksize )
{
	const double support0	 = 3.0; // Lanczos
	const double scale		 = ( double )in_size / ( double )out_size;
	const double filterscale = scale < 1.0 ? 1.0 : scale;
	const double support	 = support0 * filterscale;
	ksize					 = ( int )std::ceil( support ) * 2 + 1;

	std::vector<double> k( ( size_t )ksize );
	bounds.resize( ( size_t )out_size * 2 );
	kk.resize( ( size_t )out_size * ksize );

	for( int xx = 0; xx < out_size; ++xx ) {
		const double center = ( ( double )xx + 0.5 ) * scale;
		const double ss		= 1.0 / filterscale;
		int			 xmin	= ( int )( center - support + 0.5 );
		if( xmin < 0 )
			xmin = 0;
		int xmax = ( int )( center + support + 0.5 );
		if( xmax > in_size )
			xmax = in_size;
		xmax -= xmin;

		double ww = 0.0;
		for( int x = 0; x < xmax; ++x ) {
			const double w = pil_lanczos3( ( ( double )( x + xmin ) - center + 0.5 ) * ss );
			k[( size_t )x] = w;
			ww += w;
		}
		for( int x = 0; x < xmax; ++x ) {
			if( ww != 0.0 )
				k[( size_t )x] /= ww;
		}
		for( int x = 0; x < xmax; ++x ) {
			const double w				 = k[( size_t )x] * ( double )( 1 << PIL_PRECISION_BITS );
			kk[( size_t )xx * ksize + x] = ( int32_t )( w < 0 ? w - 0.5 : w + 0.5 );
		}
		for( int x = xmax; x < ksize; ++x )
			kk[( size_t )xx * ksize + x] = 0;
		bounds[( size_t )xx * 2 + 0] = xmin;
		bounds[( size_t )xx * 2 + 1] = xmax;
	}
}

// Separable resample of an interleaved uint8 image (any channel count),
// horizontal pass then vertical pass, PIL-compatible.
void pil_resize( const uint8_t* in, int w, int h, int ch, int out_w, int out_h, std::vector<uint8_t>& out )
{
	std::vector<int>	 bounds;
	std::vector<int32_t> kk;
	int					 ksize = 0;

	// horizontal: [h, w] -> [h, out_w]
	std::vector<uint8_t> tmp( ( size_t )h * out_w * ch );
	pil_coeffs( w, out_w, bounds, kk, ksize );
	for( int y = 0; y < h; ++y ) {
		const uint8_t* row	= in + ( size_t )y * w * ch;
		uint8_t*	   orow = tmp.data() + ( size_t )y * out_w * ch;
		for( int xx = 0; xx < out_w; ++xx ) {
			const int	   xmin = bounds[( size_t )xx * 2 + 0];
			const int	   xmax = bounds[( size_t )xx * 2 + 1];
			const int32_t* k	= kk.data() + ( size_t )xx * ksize;
			for( int c = 0; c < ch; ++c ) {
				int64_t ss = ( int64_t )1 << ( PIL_PRECISION_BITS - 1 );
				for( int x = 0; x < xmax; ++x ) {
					ss += ( int64_t )row[( size_t )( x + xmin ) * ch + c] * k[x];
				}
				orow[( size_t )xx * ch + c] = pil_clip8( ss );
			}
		}
	}

	// vertical: [h, out_w] -> [out_h, out_w]
	out.resize( ( size_t )out_h * out_w * ch );
	pil_coeffs( h, out_h, bounds, kk, ksize );
	for( int yy = 0; yy < out_h; ++yy ) {
		const int	   ymin = bounds[( size_t )yy * 2 + 0];
		const int	   ymax = bounds[( size_t )yy * 2 + 1];
		const int32_t* k	= kk.data() + ( size_t )yy * ksize;
		uint8_t*	   orow = out.data() + ( size_t )yy * out_w * ch;
		for( int xx = 0; xx < out_w; ++xx ) {
			for( int c = 0; c < ch; ++c ) {
				int64_t ss = ( int64_t )1 << ( PIL_PRECISION_BITS - 1 );
				for( int y = 0; y < ymax; ++y ) {
					ss += ( int64_t )tmp[( size_t )( y + ymin ) * out_w * ch + ( size_t )xx * ch + c] * k[y];
				}
				orow[( size_t )xx * ch + c] = pil_clip8( ss );
			}
		}
	}
}

// Python round() (banker's rounding) for the .0/.5 values PIL's crop sees.
inline int py_round_half_even( double v )
{
	const double fl	  = std::floor( v );
	const double frac = v - fl;
	if( frac < 0.5 )
		return ( int )fl;
	if( frac > 0.5 )
		return ( int )fl + 1;
	const int lo = ( int )fl;
	return ( lo % 2 == 0 ) ? lo : lo + 1;
}

} // namespace

int trellis2_remove_solid_background_rgba( uint8_t* rgba, int w, int h, int mode )
{
	if( !rgba || w <= 0 || h <= 0 || mode < TRELLIS2_BACKGROUND_AUTO || mode > TRELLIS2_BACKGROUND_WHITE ) {
		return -1;
	}
	if( mode == TRELLIS2_BACKGROUND_KEEP )
		return 0;

	const size_t count = ( size_t )w * h;
	if( mode == TRELLIS2_BACKGROUND_AUTO ) {
		// An already-masked PNG should be trusted. Requiring more than both 1%
		// and four pixels avoids treating a stray transparent metadata pixel as
		// a meaningful subject mask.
		size_t translucent = 0;
		for( size_t i = 0; i < count; ++i )
			translucent += rgba[i * 4 + 3] < 250;
		if( translucent > std::max<size_t>( 4, count / 100 ) )
			return 0;

		int	 border = 0, dark = 0, light = 0;
		auto sample = [&]( int x, int y ) {
			const uint8_t* p  = rgba + ( ( size_t )y * w + x ) * 4;
			const int	   lo = std::min( ( int )p[0], std::min( ( int )p[1], ( int )p[2] ) );
			const int	   hi = std::max( ( int )p[0], std::max( ( int )p[1], ( int )p[2] ) );
			++border;
			dark += hi <= 80;
			light += lo >= 175;
		};
		for( int x = 0; x < w; ++x ) {
			sample( x, 0 );
			if( h > 1 )
				sample( x, h - 1 );
		}
		for( int y = 1; y + 1 < h; ++y ) {
			sample( 0, y );
			if( w > 1 )
				sample( w - 1, y );
		}
		int		  dark_corners = 0, light_corners = 0;
		const int corners[4][2] = { { 0, 0 }, { w - 1, 0 }, { 0, h - 1 }, { w - 1, h - 1 } };
		for( const auto& c : corners ) {
			const uint8_t* p  = rgba + ( ( size_t )c[1] * w + c[0] ) * 4;
			const int	   lo = std::min( ( int )p[0], std::min( ( int )p[1], ( int )p[2] ) );
			const int	   hi = std::max( ( int )p[0], std::max( ( int )p[1], ( int )p[2] ) );
			dark_corners += hi <= 80;
			light_corners += lo >= 175;
		}
		const bool is_dark	= dark * 100 >= border * 55 || dark_corners >= 3;
		const bool is_light = light * 100 >= border * 55 || light_corners >= 3;
		if( !is_dark && !is_light )
			return 0;
		mode = is_dark && ( !is_light || dark >= light ) ? TRELLIS2_BACKGROUND_BLACK : TRELLIS2_BACKGROUND_WHITE;
	}

	auto distance = [&]( size_t i ) {
		const uint8_t* p = rgba + i * 4;
		if( mode == TRELLIS2_BACKGROUND_BLACK ) {
			return std::max( ( int )p[0], std::max( ( int )p[1], ( int )p[2] ) );
		}
		const int lo = std::min( ( int )p[0], std::min( ( int )p[1], ( int )p[2] ) );
		return 255 - lo;
	};
	auto				 eligible = [&]( size_t i ) { return distance( i ) <= 80; };

	std::vector<uint8_t> seen( count, 0 );
	std::vector<size_t>	 queue;
	queue.reserve( std::min<size_t>( count, ( size_t )1 << 20 ) );
	auto seed = [&]( int x, int y ) {
		const size_t i = ( size_t )y * w + x;
		if( !seen[i] && eligible( i ) ) {
			seen[i] = 1;
			queue.push_back( i );
		}
	};
	for( int x = 0; x < w; ++x ) {
		seed( x, 0 );
		if( h > 1 )
			seed( x, h - 1 );
	}
	for( int y = 1; y + 1 < h; ++y ) {
		seed( 0, y );
		if( w > 1 )
			seed( w - 1, y );
	}

	int changed = 0;
	for( size_t head = 0; head < queue.size(); ++head ) {
		const size_t i	= queue[head];
		const int	 d	= distance( i );
		float		 t	= ( float )( d - 12 ) / ( 72.0f - 12.0f );
		t				= std::max( 0.0f, std::min( 1.0f, t ) );
		t				= t * t * ( 3.0f - 2.0f * t ); // smooth feather, background -> subject
		uint8_t*	  p = rgba + i * 4;
		const uint8_t a = ( uint8_t )std::lround( ( float )p[3] * t );
		if( a != p[3] ) {
			p[3] = a;
			++changed;
		}

		const int x = ( int )( i % ( size_t )w ), y = ( int )( i / ( size_t )w );
		auto	  visit = [&]( size_t n ) {
			 if( !seen[n] && eligible( n ) ) {
				 seen[n] = 1;
				 queue.push_back( n );
			 }
		};
		if( x > 0 )
			visit( i - 1 );
		if( x + 1 < w )
			visit( i + 1 );
		if( y > 0 )
			visit( i - ( size_t )w );
		if( y + 1 < h )
			visit( i + ( size_t )w );
	}
	return changed;
}

bool trellis2_preprocess_rgba( const uint8_t* rgba, int w, int h, int out_size, std::vector<uint8_t>& out_rgb, std::string* error )
{
	if( !rgba || w <= 0 || h <= 0 || out_size <= 0 ) {
		set_error( error, "invalid arguments" );
		return false;
	}

	// 1. downscale so max(W, H) <= 1024 (PIL: int(dim * scale) floor)
	std::vector<uint8_t> img( rgba, rgba + ( size_t )w * h * 4 );
	const int			 max_size = w > h ? w : h;
	if( max_size > 1024 ) {
		const double		 scale = 1024.0 / ( double )max_size;
		const int			 nw	   = ( int )( ( double )w * scale );
		const int			 nh	   = ( int )( ( double )h * scale );
		std::vector<uint8_t> resized;
		pil_resize( img.data(), w, h, 4, nw, nh, resized );
		img = std::move( resized );
		w	= nw;
		h	= nh;
	}

	// 2. bounding box of alpha > 0.8*255, square crop centered on it
	int x0 = w, y0 = h, x1 = -1, y1 = -1;
	for( int y = 0; y < h; ++y ) {
		for( int x = 0; x < w; ++x ) {
			if( img[( ( size_t )y * w + x ) * 4 + 3] > 204 ) { // 0.8*255 = 204.0
				if( x < x0 )
					x0 = x;
				if( x > x1 )
					x1 = x;
				if( y < y0 )
					y0 = y;
				if( y > y1 )
					y1 = y;
			}
		}
	}
	if( x1 < 0 ) {
		set_error( error, "image has no pixels with alpha > 0.8 (fully transparent?)" );
		return false;
	}

	const double cx	  = ( double )( x0 + x1 ) / 2.0;
	const double cy	  = ( double )( y0 + y1 ) / 2.0;
	const int	 size = ( x1 - x0 ) > ( y1 - y0 ) ? ( x1 - x0 ) : ( y1 - y0 );
	const int	 half = size / 2;

	const int	 cx0 = py_round_half_even( cx - half );
	const int	 cy0 = py_round_half_even( cy - half );
	const int	 cx1 = py_round_half_even( cx + half );
	const int	 cy1 = py_round_half_even( cy + half );
	const int	 cw	 = cx1 - cx0;
	const int	 chh = cy1 - cy0;
	if( cw <= 0 || chh <= 0 ) {
		set_error( error, "degenerate alpha bounding box" );
		return false;
	}

	// 3. crop (zero-padded outside the source) + premultiply onto black -> RGB
	std::vector<uint8_t> rgb( ( size_t )cw * chh * 3, 0 );
	for( int y = 0; y < chh; ++y ) {
		const int sy = y + cy0;
		if( sy < 0 || sy >= h )
			continue;
		for( int x = 0; x < cw; ++x ) {
			const int sx = x + cx0;
			if( sx < 0 || sx >= w )
				continue;
			const uint8_t* p = &img[( ( size_t )sy * w + sx ) * 4];
			const float	   a = ( float )p[3] / 255.0f;
			for( int c = 0; c < 3; ++c ) {
				// matches numpy: ((rgb/255 * alpha/255) * 255).astype(uint8)
				const float v						  = ( ( float )p[c] / 255.0f ) * a * 255.0f;
				rgb[( ( size_t )y * cw + x ) * 3 + c] = ( uint8_t )v;
			}
		}
	}

	// 4. LANCZOS resize to out_size x out_size
	pil_resize( rgb.data(), cw, chh, 3, out_size, out_size, out_rgb );
	return true;
}

bool trellis2_save_dinodata( const std::string& path, const trellis2_dino_cond& cond, std::string* error )
{
	if( cond.empty() || cond.shape.empty() ) {
		set_error( error, "empty cond" );
		return false;
	}
	int64_t total = 1;
	for( int64_t d : cond.shape )
		total *= d;
	if( ( size_t )total != cond.data.size() ) {
		set_error( error, "shape/data size mismatch" );
		return false;
	}

	std::ofstream f( path, std::ios::binary );
	if( !f ) {
		set_error( error, "cannot open for writing: " + path );
		return false;
	}
	f.write( "DINOCOND", 8 );
	const uint32_t version = cond.format_version ? cond.format_version : 1;
	const uint32_t dtype   = 0; // f32
	const uint32_t ndim	   = ( uint32_t )cond.shape.size();
	auto		   w32	   = [&]( uint32_t v ) { f.write( reinterpret_cast<const char*>( &v ), 4 ); };
	w32( version );
	w32( dtype );
	w32( ndim );
	for( int64_t d : cond.shape )
		w32( ( uint32_t )d );
	f.write( reinterpret_cast<const char*>( cond.data.data() ), ( std::streamsize )( cond.data.size() * sizeof( float ) ) );
	if( !f ) {
		set_error( error, "short write: " + path );
		return false;
	}
	return true;
}

/*****************************************************************************
** Shape-SLAT flow DiT (stage 2) — GGUF loader
*****************************************************************************/

struct trellis2_slat_flow_model {
	gguf_context*								  gguf = nullptr;
	ggml_context*								  ctx  = nullptr;
	trellis2_slat_flow_hparams					  hp;
	bool										  has_data = false;

	ggml_backend_t								  backend	  = nullptr;
	ggml_backend_buffer_t						  weights_buf = nullptr;
	std::string									  backend_name;

	std::unordered_map<std::string, ggml_tensor*> tensors;
};

trellis2_slat_flow_model* trellis2_slat_flow_load( const std::string& path, bool load_tensors, std::string* error, const char* device )
{
	auto*			 m = new trellis2_slat_flow_model();

	gguf_init_params params;
	params.no_alloc = true;
	params.ctx		= &m->ctx;

	m->gguf = gguf_init_from_file( path.c_str(), params );
	if( !m->gguf ) {
		set_error( error, "gguf_init_from_file failed (not a GGUF file?): " + path );
		delete m;
		return nullptr;
	}

	const char* arch = kv_str( m->gguf, "general.architecture", "" );
	if( std::strcmp( arch, "trellis2-slat-flow" ) != 0 ) {
		set_error( error, std::string( "unexpected architecture '" ) + arch + "' (expected 'trellis2-slat-flow')" );
		trellis2_slat_flow_free( m );
		return nullptr;
	}

	trellis2_slat_flow_hparams& hp = m->hp;
	const char*					P  = "trellis2.slat_flow.";
	auto						K  = [&]( const std::string& suffix ) { return std::string( P ) + suffix; };

	hp.resolution		 = ( int32_t )kv_u32( m->gguf, K( "resolution" ).c_str(), 0 );
	hp.in_channels		 = ( int32_t )kv_u32( m->gguf, K( "in_channels" ).c_str(), 0 );
	hp.out_channels		 = ( int32_t )kv_u32( m->gguf, K( "out_channels" ).c_str(), 0 );
	hp.model_channels	 = ( int32_t )kv_u32( m->gguf, K( "model_channels" ).c_str(), 0 );
	hp.cond_channels	 = ( int32_t )kv_u32( m->gguf, K( "cond_channels" ).c_str(), 0 );
	hp.num_blocks		 = ( int32_t )kv_u32( m->gguf, K( "num_blocks" ).c_str(), 0 );
	hp.num_heads		 = ( int32_t )kv_u32( m->gguf, K( "num_heads" ).c_str(), 0 );
	hp.mlp_ratio		 = kv_f32( m->gguf, K( "mlp_ratio" ).c_str(), 0.0f );
	hp.share_mod		 = kv_bool( m->gguf, K( "share_mod" ).c_str(), false ) ? 1 : 0;
	hp.qk_rms_norm		 = kv_bool( m->gguf, K( "qk_rms_norm" ).c_str(), false ) ? 1 : 0;
	hp.qk_rms_norm_cross = kv_bool( m->gguf, K( "qk_rms_norm_cross" ).c_str(), false ) ? 1 : 0;
	hp.rope_freq_min	 = kv_f32( m->gguf, K( "rope_freq_min" ).c_str(), 1.0f );
	hp.rope_freq_base	 = kv_f32( m->gguf, K( "rope_freq_base" ).c_str(), 10000.0f );
	hp.file_type		 = ( int32_t )kv_u32( m->gguf, "general.file_type", 0 );
	std::snprintf( hp.pe_mode, sizeof( hp.pe_mode ), "%s", kv_str( m->gguf, K( "pe_mode" ).c_str(), "rope" ) );
	for( int c = 0; c < hp.out_channels && c < 64; ++c ) {
		hp.norm_mean[c] = kv_f32( m->gguf, K( "norm_mean." + std::to_string( c ) ).c_str(), 0.0f );
		hp.norm_std[c]	= kv_f32( m->gguf, K( "norm_std." + std::to_string( c ) ).c_str(), 1.0f );
	}
	// Texture SLAT flow: concat_cond. 0/absent on the shape flow. When > 0, the
	// shape SLAT (concat_norm-normalized) is concatenated onto the noise so the
	// DiT sees in_channels = out_channels + concat_cond_channels.
	hp.concat_cond_channels = ( int32_t )kv_u32( m->gguf, K( "concat_cond_channels" ).c_str(), 0 );
	for( int c = 0; c < hp.concat_cond_channels && c < 64; ++c ) {
		hp.concat_norm_mean[c] = kv_f32( m->gguf, K( "concat_norm_mean." + std::to_string( c ) ).c_str(), 0.0f );
		hp.concat_norm_std[c]  = kv_f32( m->gguf, K( "concat_norm_std." + std::to_string( c ) ).c_str(), 1.0f );
	}

	for( ggml_tensor* t = ggml_get_first_tensor( m->ctx ); t != nullptr; t = ggml_get_next_tensor( m->ctx, t ) ) {
		m->tensors[t->name] = t;
	}

	if( load_tensors ) {
		m->backend	   = init_best_backend( m->backend_name, device );
		m->weights_buf = ggml_backend_alloc_ctx_tensors( m->ctx, m->backend );
		if( !m->weights_buf ) {
			set_error( error, "failed to allocate weights on backend " + m->backend_name );
			trellis2_slat_flow_free( m );
			return nullptr;
		}
		std::ifstream fin( path, std::ios::binary );
		if( !fin ) {
			set_error( error, "cannot reopen file for weight data: " + path );
			trellis2_slat_flow_free( m );
			return nullptr;
		}
		const size_t		 data_off = gguf_get_data_offset( m->gguf );
		const int64_t		 nt		  = gguf_get_n_tensors( m->gguf );
		std::vector<uint8_t> buf;
		for( int64_t i = 0; i < nt; ++i ) {
			const char*	 name = gguf_get_tensor_name( m->gguf, i );
			ggml_tensor* t	  = m->tensors[name];
			const size_t nb	  = ggml_nbytes( t );
			const size_t off  = data_off + gguf_get_tensor_offset( m->gguf, i );
			buf.resize( nb );
			fin.seekg( ( std::streamoff )off, std::ios::beg );
			if( !fin.read( reinterpret_cast<char*>( buf.data() ), ( std::streamsize )nb ) ) {
				set_error( error, std::string( "failed reading weight '" ) + name + "' from file" );
				trellis2_slat_flow_free( m );
				return nullptr;
			}
			ggml_backend_tensor_set( t, buf.data(), 0, nb );
		}
		m->has_data = true;
	}

	return m;
}

void trellis2_slat_flow_free( trellis2_slat_flow_model* m )
{
	if( !m )
		return;
	if( m->weights_buf )
		ggml_backend_buffer_free( m->weights_buf );
	if( m->backend )
		ggml_backend_free( m->backend );
	if( m->gguf )
		gguf_free( m->gguf );
	if( m->ctx )
		ggml_free( m->ctx );
	delete m;
}

const char* trellis2_slat_flow_backend_name( const trellis2_slat_flow_model* m )
{
	return ( m && !m->backend_name.empty() ) ? m->backend_name.c_str() : "none";
}

const trellis2_slat_flow_hparams& trellis2_slat_flow_hparams_of( const trellis2_slat_flow_model* m )
{
	return m->hp;
}

/*****************************************************************************
** Shape-SLAT flow DiT — forward pass
**
** Identical block structure to trellis2_ss_flow_forward, with the dense R^3
** token grid replaced by the L active voxels: 3D RoPE phases come from each
** voxel's integer coords, everything else (shared adaLN modulation, QK-RMS
** norm, cross-attention to the DINO tokens, GELU-tanh FFN) is unchanged.
*****************************************************************************/

namespace
{

// Interleaved 3D-RoPE tables for an explicit voxel-coordinate list. Same
// layout as rope_tables(): [head_dim, 1, L] with cos[2p] == cos[2p+1].
void rope_tables_coords( const int32_t* coords, int L, int head_dim, float freq_min, float freq_base, std::vector<float>& cos_t, std::vector<float>& sin_t )
{
	const int		   dim		= 3;
	const int		   freq_dim = head_dim / 2 / dim;

	std::vector<float> freqs( ( size_t )freq_dim );
	for( int mi = 0; mi < freq_dim; ++mi ) {
		freqs[mi] = freq_min / std::pow( freq_base, ( float )mi / ( float )freq_dim );
	}

	cos_t.assign( ( size_t )head_dim * L, 1.0f );
	sin_t.assign( ( size_t )head_dim * L, 0.0f );
	const int pairs = head_dim / 2;
	for( int v = 0; v < L; ++v ) {
		for( int p = 0; p < pairs; ++p ) {
			float theta = 0.0f;
			if( p < dim * freq_dim ) {
				theta = ( float )coords[( size_t )v * 3 + p / freq_dim] * freqs[p % freq_dim];
			}
			const size_t base = ( size_t )v * head_dim + ( size_t )2 * p;
			cos_t[base] = cos_t[base + 1] = std::cos( theta );
			sin_t[base] = sin_t[base + 1] = std::sin( theta );
		}
	}
}

} // namespace

bool trellis2_slat_flow_forward(
	trellis2_slat_flow_model* m, const float* x, int n_voxels, const int32_t* coords, float t, const float* cond, int cond_tokens, int cond_channels, float* out, std::string* error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}

	const trellis2_slat_flow_hparams& hp = m->hp;
	if( std::strcmp( hp.pe_mode, "rope" ) != 0 ) {
		set_error( error, "only pe_mode=rope is implemented" );
		return false;
	}
	if( !hp.share_mod ) {
		set_error( error, "only share_mod=true is implemented" );
		return false;
	}
	if( cond_channels != hp.cond_channels ) {
		set_error( error, "cond_channels mismatch" );
		return false;
	}

	const int	C		   = hp.model_channels;
	const int	N		   = n_voxels;
	const int	H		   = hp.num_heads;
	const int	hd		   = hp.head_dim();
	const int	Lkv		   = cond_tokens;
	const float attn_scale = 1.0f / std::sqrt( ( float )hd );

	std::string missing;
	auto		W = [&]( const std::string& n ) -> ggml_tensor* {
		   auto it = m->tensors.find( n );
		   if( it == m->tensors.end() ) {
			   if( missing.empty() )
				   missing = n;
			   return nullptr;
		   }
		   return it->second;
	};

	const size_t	 mem = ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom( 32768, false );
	ggml_init_params ip { mem, nullptr, true };
	ggml_context*	 ctx = ggml_init( ip );
	ggml_cgraph*	 gf	 = ggml_new_graph_custom( ctx, 32768, false );

	ggml_tensor*	 x_t   = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, hp.in_channels, N ); // voxel-major [L][Cin]
	ggml_tensor*	 temb  = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, 256 );
	ggml_tensor*	 cos_t = ggml_new_tensor_3d( ctx, GGML_TYPE_F32, hd, 1, N );
	ggml_tensor*	 sin_t = ggml_new_tensor_3d( ctx, GGML_TYPE_F32, hd, 1, N );
	ggml_tensor*	 cnd   = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, cond_channels, Lkv );
	ggml_set_input( x_t );
	ggml_set_input( temb );
	ggml_set_input( cos_t );
	ggml_set_input( sin_t );
	ggml_set_input( cnd );

	auto lin = [&]( ggml_tensor* in, const std::string& pfx ) -> ggml_tensor* {
		ggml_tensor* y = ggml_mul_mat( ctx, W( pfx + ".weight" ), in );
		ggml_tensor* b = W( pfx + ".bias" );
		if( b )
			y = ggml_add( ctx, y, b );
		return y;
	};
	auto modulate = [&]( ggml_tensor* h, ggml_tensor* scale, ggml_tensor* shift ) { return ggml_add( ctx, ggml_add( ctx, ggml_mul( ctx, h, scale ), h ), shift ); };
	auto rope	  = [&]( ggml_tensor* q3 ) -> ggml_tensor* {
		ggml_tensor* q4	  = ggml_reshape_4d( ctx, q3, 2, hd / 2, H, N );
		ggml_tensor* q0	  = ggml_cont( ctx, ggml_view_4d( ctx, q4, 1, hd / 2, H, N, q4->nb[1], q4->nb[2], q4->nb[3], 0 ) );
		ggml_tensor* q1	  = ggml_cont( ctx, ggml_view_4d( ctx, q4, 1, hd / 2, H, N, q4->nb[1], q4->nb[2], q4->nb[3], q4->nb[0] ) );
		ggml_tensor* swap = ggml_concat( ctx, ggml_neg( ctx, q1 ), q0, 0 );
		swap			  = ggml_reshape_3d( ctx, swap, hd, H, N );
		return ggml_add( ctx, ggml_mul( ctx, q3, cos_t ), ggml_mul( ctx, swap, sin_t ) );
	};
	auto		 qk_norm = [&]( ggml_tensor* v3, const std::string& gname ) { return ggml_mul( ctx, ggml_rms_norm( ctx, v3, 1e-12f ), W( gname ) ); };
	auto		 sdpa	 = [&]( ggml_tensor* q3, ggml_tensor* k3, ggml_tensor* v3 ) { return sdpa_auto( ctx, q3, k3, v3, C, attn_scale ); };

	const size_t es = sizeof( float );

	ggml_tensor* h = lin( x_t, "input_layer" ); // [C, N]

	ggml_tensor* te	  = lin( temb, "t_embedder.mlp.0" );
	te				  = ggml_silu( ctx, te );
	te				  = lin( te, "t_embedder.mlp.2" );
	ggml_tensor* tmod = lin( ggml_silu( ctx, te ), "adaLN_modulation.1" );

	ggml_tensor* cond_h = cnd;

	for( int b = 0; b < hp.num_blocks; ++b ) {
		const std::string blk		= "blocks." + std::to_string( b );
		ggml_tensor*	  mods		= ggml_add( ctx, W( blk + ".modulation" ), tmod );
		auto			  chunk		= [&]( int idx ) { return ggml_view_1d( ctx, mods, C, ( size_t )idx * C * es ); };
		ggml_tensor *	  shift_msa = chunk( 0 ), *scale_msa = chunk( 1 ), *gate_msa = chunk( 2 );
		ggml_tensor *	  shift_mlp = chunk( 3 ), *scale_mlp = chunk( 4 ), *gate_mlp = chunk( 5 );

		ggml_tensor*	  hn  = modulate( ggml_norm( ctx, h, 1e-6f ), scale_msa, shift_msa );
		ggml_tensor*	  qkv = lin( hn, blk + ".self_attn.to_qkv" );
		ggml_tensor*	  q	  = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, qkv, C, N, qkv->nb[1], 0 ) ), hd, H, N );
		ggml_tensor*	  k	  = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, qkv, C, N, qkv->nb[1], ( size_t )C * es ) ), hd, H, N );
		ggml_tensor*	  v	  = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, qkv, C, N, qkv->nb[1], ( size_t )2 * C * es ) ), hd, H, N );
		q					  = rope( qk_norm( q, blk + ".self_attn.q_rms_norm.gamma" ) );
		k					  = rope( qk_norm( k, blk + ".self_attn.k_rms_norm.gamma" ) );
		ggml_tensor* sa		  = lin( sdpa( q, k, v ), blk + ".self_attn.to_out" );
		h					  = ggml_add( ctx, h, ggml_mul( ctx, sa, gate_msa ) );

		ggml_tensor* h2 = ggml_norm( ctx, h, 1e-6f );
		h2				= ggml_add( ctx, ggml_mul( ctx, h2, W( blk + ".norm2.weight" ) ), W( blk + ".norm2.bias" ) );
		ggml_tensor* cq = ggml_reshape_3d( ctx, lin( h2, blk + ".cross_attn.to_q" ), hd, H, N );
		cq				= qk_norm( cq, blk + ".cross_attn.q_rms_norm.gamma" );
		ggml_tensor* kv = lin( cond_h, blk + ".cross_attn.to_kv" );
		ggml_tensor* ck = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, kv, C, Lkv, kv->nb[1], 0 ) ), hd, H, Lkv );
		ggml_tensor* cv = ggml_reshape_3d( ctx, ggml_cont( ctx, ggml_view_2d( ctx, kv, C, Lkv, kv->nb[1], ( size_t )C * es ) ), hd, H, Lkv );
		ck				= qk_norm( ck, blk + ".cross_attn.k_rms_norm.gamma" );
		ggml_tensor* ca = lin( sdpa( cq, ck, cv ), blk + ".cross_attn.to_out" );
		h				= ggml_add( ctx, h, ca );

		ggml_tensor* hm = modulate( ggml_norm( ctx, h, 1e-6f ), scale_mlp, shift_mlp );
		hm				= lin( hm, blk + ".mlp.mlp.0" );
		hm				= ggml_gelu( ctx, hm );
		hm				= lin( hm, blk + ".mlp.mlp.2" );
		h				= ggml_add( ctx, h, ggml_mul( ctx, hm, gate_mlp ) );
	}

	h			   = ggml_norm( ctx, h, 1e-5f );
	h			   = lin( h, "out_layer" ); // [Cout, N]
	ggml_tensor* y = ggml_cont( ctx, h );	// voxel-major [L][Cout]
	ggml_set_output( y );

	if( !missing.empty() ) {
		set_error( error, "missing tensor: " + missing );
		ggml_free( ctx );
		return false;
	}

	ggml_build_forward_expand( gf, y );

	ggml_backend_t backend = m->backend;
	ggml_gallocr_t alloc   = ggml_gallocr_new( ggml_backend_get_default_buffer_type( backend ) );
	if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
		set_error( error, "ggml_gallocr_alloc_graph failed" );
		ggml_gallocr_free( alloc );
		ggml_free( ctx );
		return false;
	}

	std::vector<float> emb = timestep_embedding( t, 256 );
	std::vector<float> cosv, sinv;
	rope_tables_coords( coords, N, hd, hp.rope_freq_min, hp.rope_freq_base, cosv, sinv );

	ggml_backend_tensor_set( x_t, x, 0, ( size_t )hp.in_channels * N * es );
	ggml_backend_tensor_set( temb, emb.data(), 0, emb.size() * es );
	ggml_backend_tensor_set( cos_t, cosv.data(), 0, cosv.size() * es );
	ggml_backend_tensor_set( sin_t, sinv.data(), 0, sinv.size() * es );
	ggml_backend_tensor_set( cnd, cond, 0, ( size_t )cond_channels * Lkv * es );

	const ggml_status st = ggml_backend_graph_compute( backend, gf );
	bool			  ok = ( st == GGML_STATUS_SUCCESS );
	if( ok ) {
		ggml_backend_tensor_get( y, out, 0, ( size_t )hp.out_channels * N * es );
	} else {
		set_error( error, "graph compute failed" );
	}

	ggml_gallocr_free( alloc );
	ggml_free( ctx );
	return ok;
}

bool trellis2_slat_flow_sample( trellis2_slat_flow_model* m,
	int													  n_voxels,
	const int32_t*										  coords,
	const float*										  cond,
	int													  cond_tokens,
	int													  cond_channels,
	const trellis2_ss_sampler_params*					  params_in,
	const float*										  noise,
	bool												  denormalize,
	float*												  out_latent,
	std::string*										  error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}

	trellis2_ss_sampler_params P;
	if( params_in )
		P = *params_in;

	const trellis2_slat_flow_hparams& hp = m->hp;
	const size_t					  n	 = ( size_t )hp.in_channels * n_voxels;
	const double					  sm = P.sigma_min;

	std::vector<float>				  x_t( n );
	if( noise ) {
		std::memcpy( x_t.data(), noise, n * sizeof( float ) );
	} else {
		std::mt19937_64					rng( P.seed );
		std::normal_distribution<float> nd( 0.0f, 1.0f );
		for( size_t i = 0; i < n; ++i )
			x_t[i] = nd( rng );
	}

	std::vector<double> ts( ( size_t )P.steps + 1 );
	for( int i = 0; i <= P.steps; ++i ) {
		const double lin = 1.0 - ( double )i / ( double )P.steps;
		ts[i]			 = P.rescale_t * lin / ( 1.0 + ( P.rescale_t - 1.0 ) * lin );
	}

	const std::vector<float> zero_cond( ( size_t )cond_tokens * cond_channels, 0.0f );
	std::vector<float>		 pred_pos( n ), pred_neg( n ), pred_v( n ), x0_pos( n ), x0_cfg( n );

	auto					 fwd = [&]( double t, const float* c, std::vector<float>& dst ) -> bool {
		return trellis2_slat_flow_forward( m, x_t.data(), n_voxels, coords, ( float )( 1000.0 * t ), c, cond_tokens, cond_channels, dst.data(), error );
	};

	for( int i = 0; i < P.steps; ++i ) {
		const double t = ts[i], t_prev = ts[i + 1];
		const bool	 in_interval = ( t >= P.guidance_interval_min && t <= P.guidance_interval_max );
		const float	 gs			 = in_interval ? P.guidance_strength : 1.0f;

		if( gs == 1.0f ) {
			if( !fwd( t, cond, pred_v ) )
				return false;
		} else if( gs == 0.0f ) {
			if( !fwd( t, zero_cond.data(), pred_v ) )
				return false;
		} else {
			if( !fwd( t, cond, pred_pos ) )
				return false;
			if( !fwd( t, zero_cond.data(), pred_neg ) )
				return false;
			for( size_t k = 0; k < n; ++k )
				pred_v[k] = gs * pred_pos[k] + ( 1.0f - gs ) * pred_neg[k];

			if( P.guidance_rescale > 0.0f ) {
				pred_to_xstart( x_t, t, sm, pred_pos, x0_pos );
				pred_to_xstart( x_t, t, sm, pred_v, x0_cfg );
				const double std_pos = unbiased_std( x0_pos );
				const double std_cfg = unbiased_std( x0_cfg );
				const double ratio	 = ( std_cfg != 0.0 ) ? std_pos / std_cfg : 1.0;
				const float	 gr		 = P.guidance_rescale;
				for( size_t k = 0; k < n; ++k ) {
					const double rescaled = x0_cfg[k] * ratio;
					x0_cfg[k]			  = ( float )( gr * rescaled + ( 1.0 - gr ) * x0_cfg[k] );
				}
				xstart_to_pred( x_t, t, sm, x0_cfg, pred_v );
			}
		}

		const double dt = t - t_prev;
		for( size_t k = 0; k < n; ++k )
			x_t[k] = ( float )( x_t[k] - dt * pred_v[k] );

		if( P.verbose ) {
			std::fprintf( stderr, "\r[slat sample] step %2d/%d  t=%.4f->%.4f  %s   ", i + 1, P.steps, t, t_prev, in_interval ? "cfg" : "uncond" );
			std::fflush( stderr );
		}
		if( P.progress )
			P.progress( P.progress_user, i + 1, P.steps );
	}
	if( P.verbose )
		std::fprintf( stderr, "\n" );

	if( denormalize ) {
		const int C = hp.in_channels;
		for( int v = 0; v < n_voxels; ++v ) {
			for( int c = 0; c < C; ++c ) {
				x_t[( size_t )v * C + c] = x_t[( size_t )v * C + c] * hp.norm_std[c] + hp.norm_mean[c];
			}
		}
	}

	std::memcpy( out_latent, x_t.data(), n * sizeof( float ) );
	return true;
}

// Texture-SLAT flow sampling with concat_cond. Same flow-Euler loop, but the
// diffused variable is out_channels (32) and each forward is fed a fresh
// [noise(32) | normalized shape-SLAT(32)] = in_channels (64) input.
bool trellis2_slat_flow_sample_tex( trellis2_slat_flow_model* m,
	int														  n_voxels,
	const int32_t*											  coords,
	const float*											  cond,
	int														  cond_tokens,
	int														  cond_channels,
	const float*											  shape_slat,
	const trellis2_ss_sampler_params*						  params_in,
	const float*											  noise,
	bool													  denormalize,
	float*													  out_latent,
	std::string*											  error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}

	const trellis2_slat_flow_hparams& hp  = m->hp;
	const int						  Cin = hp.in_channels, Cout = hp.out_channels, Ccat = hp.concat_cond_channels;
	if( Ccat <= 0 || Cin != Cout + Ccat ) {
		set_error( error, "model is not a concat_cond (texture) flow" );
		return false;
	}

	trellis2_ss_sampler_params P;
	if( params_in )
		P = *params_in;
	const size_t	   n  = ( size_t )Cout * n_voxels; // diffused variable
	const double	   sm = P.sigma_min;

	std::vector<float> x_t( n );
	if( noise ) {
		std::memcpy( x_t.data(), noise, n * sizeof( float ) );
	} else {
		std::mt19937_64					rng( P.seed );
		std::normal_distribution<float> nd( 0.0f, 1.0f );
		for( size_t i = 0; i < n; ++i )
			x_t[i] = nd( rng );
	}

	// shape SLAT normalized by concat_norm (once); concatenated onto the noise.
	std::vector<float> shape_n( ( size_t )Ccat * n_voxels );
	for( int v = 0; v < n_voxels; ++v )
		for( int c = 0; c < Ccat; ++c )
			shape_n[( size_t )v * Ccat + c] = ( shape_slat[( size_t )v * Ccat + c] - hp.concat_norm_mean[c] ) / hp.concat_norm_std[c];

	std::vector<double> ts( ( size_t )P.steps + 1 );
	for( int i = 0; i <= P.steps; ++i ) {
		const double lin = 1.0 - ( double )i / ( double )P.steps;
		ts[i]			 = P.rescale_t * lin / ( 1.0 + ( P.rescale_t - 1.0 ) * lin );
	}

	const std::vector<float> zero_cond( ( size_t )cond_tokens * cond_channels, 0.0f );
	std::vector<float>		 pred_pos( n ), pred_neg( n ), pred_v( n ), x0_pos( n ), x0_cfg( n );
	std::vector<float>		 xin( ( size_t )Cin * n_voxels );

	auto					 fwd = [&]( double t, const float* c, std::vector<float>& dst ) -> bool {
		for( int v = 0; v < n_voxels; ++v ) {
			float* d = xin.data() + ( size_t )v * Cin;
			std::memcpy( d, x_t.data() + ( size_t )v * Cout, ( size_t )Cout * sizeof( float ) );
			std::memcpy( d + Cout, shape_n.data() + ( size_t )v * Ccat, ( size_t )Ccat * sizeof( float ) );
		}
		return trellis2_slat_flow_forward( m, xin.data(), n_voxels, coords, ( float )( 1000.0 * t ), c, cond_tokens, cond_channels, dst.data(), error );
	};

	for( int i = 0; i < P.steps; ++i ) {
		const double t = ts[i], t_prev = ts[i + 1];
		const bool	 in_interval = ( t >= P.guidance_interval_min && t <= P.guidance_interval_max );
		const float	 gs			 = in_interval ? P.guidance_strength : 1.0f;

		if( gs == 1.0f ) {
			if( !fwd( t, cond, pred_v ) )
				return false;
		} else if( gs == 0.0f ) {
			if( !fwd( t, zero_cond.data(), pred_v ) )
				return false;
		} else {
			if( !fwd( t, cond, pred_pos ) )
				return false;
			if( !fwd( t, zero_cond.data(), pred_neg ) )
				return false;
			for( size_t k = 0; k < n; ++k )
				pred_v[k] = gs * pred_pos[k] + ( 1.0f - gs ) * pred_neg[k];

			if( P.guidance_rescale > 0.0f ) {
				pred_to_xstart( x_t, t, sm, pred_pos, x0_pos );
				pred_to_xstart( x_t, t, sm, pred_v, x0_cfg );
				const double std_pos = unbiased_std( x0_pos );
				const double std_cfg = unbiased_std( x0_cfg );
				const double ratio	 = ( std_cfg != 0.0 ) ? std_pos / std_cfg : 1.0;
				const float	 gr		 = P.guidance_rescale;
				for( size_t k = 0; k < n; ++k ) {
					const double rescaled = x0_cfg[k] * ratio;
					x0_cfg[k]			  = ( float )( gr * rescaled + ( 1.0 - gr ) * x0_cfg[k] );
				}
				xstart_to_pred( x_t, t, sm, x0_cfg, pred_v );
			}
		}

		const double dt = t - t_prev;
		for( size_t k = 0; k < n; ++k )
			x_t[k] = ( float )( x_t[k] - dt * pred_v[k] );

		if( P.verbose ) {
			std::fprintf( stderr, "\r[tex slat sample] step %2d/%d  t=%.4f->%.4f  %s   ", i + 1, P.steps, t, t_prev, in_interval ? "cfg" : "uncond" );
			std::fflush( stderr );
		}
		if( P.progress )
			P.progress( P.progress_user, i + 1, P.steps );
	}
	if( P.verbose )
		std::fprintf( stderr, "\n" );

	if( denormalize ) {
		for( int v = 0; v < n_voxels; ++v )
			for( int c = 0; c < Cout; ++c )
				x_t[( size_t )v * Cout + c] = x_t[( size_t )v * Cout + c] * hp.norm_std[c] + hp.norm_mean[c];
	}

	std::memcpy( out_latent, x_t.data(), n * sizeof( float ) );
	return true;
}

/*****************************************************************************
** Shape-SLAT VAE decoder (FlexiDualGridVaeDecoder) — GGUF loader
*****************************************************************************/

struct trellis2_shape_dec_model {
	gguf_context*								  gguf = nullptr;
	ggml_context*								  ctx  = nullptr;
	trellis2_shape_dec_hparams					  hp;
	bool										  has_data = false;

	ggml_backend_t								  backend	  = nullptr;
	ggml_backend_buffer_t						  weights_buf = nullptr;
	std::string									  backend_name;

	std::unordered_map<std::string, ggml_tensor*> tensors;
};

static trellis2_shape_dec_model* dec_load_impl( const std::string& path, bool load_tensors, std::string* error, const char* device, const char* expect_arch, const char* kv_prefix )
{
	auto*			 m = new trellis2_shape_dec_model();

	gguf_init_params params;
	params.no_alloc = true;
	params.ctx		= &m->ctx;

	m->gguf = gguf_init_from_file( path.c_str(), params );
	if( !m->gguf ) {
		set_error( error, "gguf_init_from_file failed (not a GGUF file?): " + path );
		delete m;
		return nullptr;
	}

	const char* arch = kv_str( m->gguf, "general.architecture", "" );
	if( std::strcmp( arch, expect_arch ) != 0 ) {
		set_error( error, std::string( "unexpected architecture '" ) + arch + "' (expected '" + expect_arch + "')" );
		trellis2_shape_dec_free( m );
		return nullptr;
	}

	trellis2_shape_dec_hparams& hp = m->hp;
	const char*					P  = kv_prefix;
	auto						K  = [&]( const std::string& suffix ) { return std::string( P ) + suffix; };

	hp.latent_channels = ( int32_t )kv_u32( m->gguf, K( "latent_channels" ).c_str(), 0 );
	hp.out_channels	   = ( int32_t )kv_u32( m->gguf, K( "out_channels" ).c_str(), 0 );
	hp.n_levels		   = ( int32_t )kv_u32( m->gguf, K( "n_levels" ).c_str(), 0 );
	hp.norm_eps		   = kv_f32( m->gguf, K( "norm_eps" ).c_str(), 1e-6f );
	hp.voxel_margin	   = kv_f32( m->gguf, K( "voxel_margin" ).c_str(), 0.5f );
	hp.file_type	   = ( int32_t )kv_u32( m->gguf, "general.file_type", 0 );
	for( int i = 0; i < hp.n_levels && i < 8; ++i ) {
		hp.channels[i]	 = ( int32_t )kv_u32( m->gguf, K( "channels." + std::to_string( i ) ).c_str(), 0 );
		hp.num_blocks[i] = ( int32_t )kv_u32( m->gguf, K( "num_blocks." + std::to_string( i ) ).c_str(), 0 );
	}

	for( ggml_tensor* t = ggml_get_first_tensor( m->ctx ); t != nullptr; t = ggml_get_next_tensor( m->ctx, t ) ) {
		m->tensors[t->name] = t;
	}

	if( load_tensors ) {
		m->backend	   = init_best_backend( m->backend_name, device );
		m->weights_buf = ggml_backend_alloc_ctx_tensors( m->ctx, m->backend );
		if( !m->weights_buf ) {
			set_error( error, "failed to allocate weights on backend " + m->backend_name );
			trellis2_shape_dec_free( m );
			return nullptr;
		}
		std::ifstream fin( path, std::ios::binary );
		if( !fin ) {
			set_error( error, "cannot reopen file for weight data: " + path );
			trellis2_shape_dec_free( m );
			return nullptr;
		}
		const size_t		 data_off = gguf_get_data_offset( m->gguf );
		const int64_t		 nt		  = gguf_get_n_tensors( m->gguf );
		std::vector<uint8_t> buf;
		for( int64_t i = 0; i < nt; ++i ) {
			const char*	 name = gguf_get_tensor_name( m->gguf, i );
			ggml_tensor* t	  = m->tensors[name];
			const size_t nb	  = ggml_nbytes( t );
			const size_t off  = data_off + gguf_get_tensor_offset( m->gguf, i );
			buf.resize( nb );
			fin.seekg( ( std::streamoff )off, std::ios::beg );
			if( !fin.read( reinterpret_cast<char*>( buf.data() ), ( std::streamsize )nb ) ) {
				set_error( error, std::string( "failed reading weight '" ) + name + "' from file" );
				trellis2_shape_dec_free( m );
				return nullptr;
			}
			ggml_backend_tensor_set( t, buf.data(), 0, nb );
		}
		m->has_data = true;
	}

	return m;
}

trellis2_shape_dec_model* trellis2_shape_dec_load( const std::string& path, bool load_tensors, std::string* error, const char* device )
{
	return dec_load_impl( path, load_tensors, error, device, "trellis2-shape-dec", "trellis2.shape_dec." );
}

// The texture decoder is the same struct/driver as the shape decoder (out=6,
// pred_subdiv=False supplied at decode time via trellis2_tex_dec_decode).
trellis2_shape_dec_model* trellis2_tex_dec_load( const std::string& path, bool load_tensors, std::string* error, const char* device )
{
	return dec_load_impl( path, load_tensors, error, device, "trellis2-tex-dec", "trellis2.tex_dec." );
}

void trellis2_shape_dec_free( trellis2_shape_dec_model* m )
{
	if( !m )
		return;
	if( m->weights_buf )
		ggml_backend_buffer_free( m->weights_buf );
	if( m->backend )
		ggml_backend_free( m->backend );
	if( m->gguf )
		gguf_free( m->gguf );
	if( m->ctx )
		ggml_free( m->ctx );
	delete m;
}

const char* trellis2_shape_dec_backend_name( const trellis2_shape_dec_model* m )
{
	return ( m && !m->backend_name.empty() ) ? m->backend_name.c_str() : "none";
}

const trellis2_shape_dec_hparams& trellis2_shape_dec_hparams_of( const trellis2_shape_dec_model* m )
{
	return m->hp;
}

/*****************************************************************************
** Shape-SLAT VAE decoder — forward
**
** Mirrors SparseUnetVaeDecoder.forward level by level. Submanifold sparse
** 3x3x3 convolutions are expressed as 27 x (get_rows gather + GEMM): the
** neighbor row index of every voxel for each kernel offset is precomputed on
** the host with a hash map (missing neighbors point at an appended zero row).
** The subdivision decision of each up-block crosses the device boundary (its
** logits pick which children exist), so each level runs as its own graph and
** the feature matrix round-trips through host memory between levels.
*****************************************************************************/

namespace
{

// key for a voxel coordinate (10 bits per axis is plenty: res <= 1024)
inline uint64_t voxel_key( int32_t c1, int32_t c2, int32_t c3 )
{
	return ( ( uint64_t )( uint32_t )c1 << 40 ) | ( ( uint64_t )( uint32_t )c2 << 20 ) | ( uint64_t )( uint32_t )c3;
}

// Neighbor row indices for all 27 offsets of a 3^3 submanifold conv.
// idx[k][v] = row of voxel v's neighbor at offset k, or L (the zero row).
// Kernel flattening matches the [Co, kD, kH, kW, Ci] weight layout with
// kD -> c1, kH -> c2, kW -> c3.
void build_neighbor_indices( const std::vector<int32_t>& coords, int L, std::vector<std::vector<int32_t>>& idx )
{
	std::unordered_map<uint64_t, int32_t> map;
	map.reserve( ( size_t )L * 2 );
	for( int v = 0; v < L; ++v ) {
		map[voxel_key( coords[( size_t )v * 3], coords[( size_t )v * 3 + 1], coords[( size_t )v * 3 + 2] )] = v;
	}
	idx.assign( 27, std::vector<int32_t>( ( size_t )L ) );
	for( int k = 0; k < 27; ++k ) {
		const int			  d1 = k / 9 - 1, d2 = ( k / 3 ) % 3 - 1, d3 = k % 3 - 1;
		std::vector<int32_t>& ik = idx[k];
		for( int v = 0; v < L; ++v ) {
			const int32_t c1 = coords[( size_t )v * 3] + d1;
			const int32_t c2 = coords[( size_t )v * 3 + 1] + d2;
			const int32_t c3 = coords[( size_t )v * 3 + 2] + d3;
			if( c1 < 0 || c2 < 0 || c3 < 0 ) {
				ik[v] = L;
				continue;
			}
			auto it = map.find( voxel_key( c1, c2, c3 ) );
			ik[v]	= ( it == map.end() ) ? L : it->second;
		}
	}
}

} // namespace

// ggml_mul_mat on the CUDA/HIP backend silently stops writing its result past a
// fixed column count — 2^21 columns with f16 weights, 2^19 with f32 (measured by
// tests/test_large_rows). Past the limit the output stays 0 and the graph still
// reports GGML_STATUS_SUCCESS. That is what broke the 1024^3 cascade: its finest
// level carries ~2.9M voxels, so every voxel past 2^21 came out of conv() as the
// bias alone (a dead accumulator plus b), the dual grid found no edge crossings
// there, and the mesh lost every face in that slab. The two functions below keep
// every graph clear of that cliff.

// How many columns one mul_mat survives, by weight type: f16 and bf16 both reach
// 2^21, f32 only 2^19 (the GEMM tiles the column dimension differently per type).
// All three are measured by tests/test_large_rows. Quantized weights take yet
// another kernel path and are not, so they get the stricter bound. This is a
// ceiling on what a single graph can compute, not a target.
static int64_t mul_mat_max_rows( ggml_type wt )
{
	return ( wt == GGML_TYPE_F16 || wt == GGML_TYPE_BF16 ) ? ( int64_t )1 << 21 : ( int64_t )1 << 19;
}

// Voxels per block for the levels that can be split, kept well under either
// ceiling. TRELLIS2_CHUNK_ROWS forces a smaller block so a test can exercise the
// block-wise path on a small model; re-read per call rather than cached, so one
// process can decode the same input both ways (tests/test_chunked_decode).
static int64_t chunk_rows_limit()
{
	if( const char* e = std::getenv( "TRELLIS2_CHUNK_ROWS" ) ) {
		const long long n = std::atoll( e );
		if( n > 0 )
			return ( int64_t )n;
	}
	return ( int64_t )1 << 18; // 262144
}

// The finest decoder level, evaluated in voxel blocks so no mul_mat ever sees
// more columns than the backend can write.
//
// This is the only level that outgrows the limit in practice, and the only one
// that splits without materialising a full-length intermediate: num_blocks is 0
// at the finest level for both shipped decoders, so the level is a single
// submanifold conv wrapped in per-row ops. norm and silu are per-row, so they
// commute with the neighbour gather — gathering raw rows and normalising after
// is exactly what the unchunked graph computes, including for the clamped
// missing-neighbour rows, which the mask zeroes either way. Only the gather
// source has to see all L rows, so it stays resident on the device for the whole
// pass while each block allocates just its own slice.
static bool shape_dec_final_level_chunked( trellis2_shape_dec_model* m,
	const trellis2_shape_dec_hparams&								 hp,
	int																 lvl,
	int																 C,
	int																 prev_C,
	int																 L,
	const std::vector<std::vector<int32_t>>&						 nidx,
	const std::vector<float>&										 up_hch,
	const std::vector<float>&										 up_xch,
	bool															 pbr_scale,
	int64_t															 chunk,
	std::vector<float>&												 out_feats,
	std::string*													 error )
{
	const size_t	  es = sizeof( float );
	const int		  xC = prev_C / 8; // skip channels
	const int		  r	 = C / xC;	   // repeat_interleave factor
	const int		  OC = hp.out_channels;
	const std::string up = "blocks." + std::to_string( lvl - 1 ) + "." + std::to_string( hp.num_blocks[lvl - 1] );

	auto			  W = [&]( const std::string& n ) -> ggml_tensor* {
		 auto it = m->tensors.find( n );
		 return it == m->tensors.end() ? nullptr : it->second;
	};
	ggml_tensor* cw = W( up + ".conv2.weight" ); // [Ci, 27, Co]
	ggml_tensor* cb = W( up + ".conv2.bias" );
	ggml_tensor* ow = W( "output_layer.weight" );
	ggml_tensor* ob = W( "output_layer.bias" );
	if( !cw || !cb || !ow ) {
		set_error( error, "missing tensor for the chunked final level" );
		return false;
	}

	// Resident level input: the neighbour gather reads arbitrary rows, so this
	// cannot be sliced per block, and re-uploading it per block would move
	// hundreds of MB across the bus for nothing.
	ggml_init_params pip { ggml_tensor_overhead() * 4, nullptr, true };
	ggml_context*	 pctx = ggml_init( pip );
	if( !pctx ) {
		set_error( error, "ggml_init failed (resident level input)" );
		return false;
	}
	ggml_tensor*		  hch_full = ggml_new_tensor_2d( pctx, GGML_TYPE_F32, C, L );
	ggml_backend_buffer_t pbuf	   = ggml_backend_alloc_ctx_tensors( pctx, m->backend );
	if( !pbuf ) {
		set_error( error, "failed to allocate the resident level input (" + std::to_string( ( size_t )C * L * es >> 20 ) + " MiB)" );
		ggml_free( pctx );
		return false;
	}
	ggml_backend_tensor_set( hch_full, up_hch.data(), 0, up_hch.size() * es );

	out_feats.resize( ( size_t )OC * L );

	const int64_t Ci = cw->ne[0], Co = cw->ne[2];
	bool		  ok = true;

	for( int64_t v0 = 0; v0 < L && ok; v0 += chunk ) {
		const int				  n = ( int )std::min<int64_t>( chunk, ( int64_t )L - v0 );

		const size_t			  gsize = 512;
		ggml_init_params		  ip { ggml_tensor_overhead() * gsize + ggml_graph_overhead_custom( gsize, false ), nullptr, true };
		ggml_context*			  ctx = ggml_init( ip );
		ggml_cgraph*			  gf  = ggml_new_graph_custom( ctx, gsize, false );

		std::vector<ggml_tensor*> idx_t( 27 ), mask_t( 27 );
		for( int k = 0; k < 27; ++k ) {
			idx_t[k]  = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, n );
			mask_t[k] = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, n );
			ggml_set_input( idx_t[k] );
			ggml_set_input( mask_t[k] );
		}
		ggml_tensor* xch = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, xC, n );
		ggml_set_input( xch );

		// conv2 with norm/silu pulled behind the gather (see the note above).
		ggml_tensor* acc = nullptr;
		for( int k = 0; k < 27; ++k ) {
			ggml_tensor* wk = ggml_cont( ctx, ggml_view_3d( ctx, cw, Ci, 1, Co, cw->nb[1], cw->nb[2], ( size_t )k * cw->nb[1] ) );
			wk				= ggml_reshape_2d( ctx, wk, Ci, Co );
			ggml_tensor* g	= ggml_get_rows( ctx, hch_full, idx_t[k] ); // [C, n]
			g				= ggml_norm( ctx, g, hp.norm_eps );
			g				= ggml_silu( ctx, g );
			g				= ggml_mul( ctx, g, mask_t[k] ); // zero missing
			ggml_tensor* y	= ggml_mul_mat( ctx, wk, g );	 // [C_out, n]
			acc				= acc ? ggml_add( ctx, acc, y ) : y;
		}
		acc = ggml_add( ctx, acc, cb );

		ggml_tensor* skip = ggml_reshape_3d( ctx, xch, 1, xC, n );
		skip			  = ggml_repeat( ctx, skip, ggml_new_tensor_3d( ctx, GGML_TYPE_F32, r, xC, n ) );
		skip			  = ggml_reshape_2d( ctx, skip, C, n );

		ggml_tensor* h = ggml_add( ctx, acc, skip );
		h			   = ggml_norm( ctx, h, 1e-5f ); // final affine-free LN
		ggml_tensor* o = ggml_mul_mat( ctx, ow, h ); // output_layer
		if( ob )
			o = ggml_add( ctx, o, ob );
		o = ggml_cont( ctx, o ); // [OC, n]
		ggml_set_output( o );
		ggml_build_forward_expand( gf, o );

		ggml_gallocr_t alloc = ggml_gallocr_new( ggml_backend_get_default_buffer_type( m->backend ) );
		if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
			set_error( error, "ggml_gallocr_alloc_graph failed (final level, voxels from " + std::to_string( v0 ) + ")" );
			ok = false;
		} else {
			std::vector<int32_t> clamped( ( size_t )n );
			std::vector<float>	 mask( ( size_t )n );
			for( int k = 0; k < 27; ++k ) {
				const std::vector<int32_t>& ik = nidx[k];
				for( int i = 0; i < n; ++i ) {
					const int32_t nb	 = ik[( size_t )( v0 + i )];
					const bool	  miss	 = nb >= L;
					clamped[( size_t )i] = miss ? 0 : nb;
					mask[( size_t )i]	 = miss ? 0.0f : 1.0f;
				}
				ggml_backend_tensor_set( idx_t[k], clamped.data(), 0, ( size_t )n * sizeof( int32_t ) );
				ggml_backend_tensor_set( mask_t[k], mask.data(), 0, ( size_t )n * sizeof( float ) );
			}
			ggml_backend_tensor_set( xch, up_xch.data() + ( size_t )v0 * xC, 0, ( size_t )xC * n * es );

			if( ggml_backend_graph_compute( m->backend, gf ) != GGML_STATUS_SUCCESS ) {
				set_error( error, "graph compute failed (final level, voxels from " + std::to_string( v0 ) + ")" );
				ok = false;
			} else {
				ggml_backend_tensor_get( o, out_feats.data() + ( size_t )v0 * OC, 0, ( size_t )OC * n * es );
			}
		}
		ggml_gallocr_free( alloc );
		ggml_free( ctx );
	}

	ggml_backend_buffer_free( pbuf );
	ggml_free( pctx );

	if( ok && pbr_scale ) // tex decoder: map to [0,1] like the reference *0.5+0.5
		for( float& f : out_feats )
			f = f * 0.5f + 0.5f;
	return ok;
}

// An intermediate decoder level — ConvNeXt blocks plus an up-block — evaluated
// in voxel blocks, so no single graph ever sees more columns than the backend
// can write.
//
// shape_dec_final_level_chunked only covers the finest level, which is one conv
// wrapped in per-row ops. A level with chained ConvNeXt blocks cannot be split
// the same way: block b+1 reads the NEIGHBOURS of block b's output, which a
// per-block graph does not have. The way out is not a halo — with a dozen
// blocks that would cost more than the level — but the same move the finest
// level already makes, keeping the input resident over all L and gathering from
// it, applied once per STAGE instead of once per level:
//
//   stage A     y = conv2( silu( LN_free( hch ) ) )        [C, L]
//   host        h = y + repeat_interleave( xch )
//   stage B_b   h = convnext_b( h )                        [C, L], per block b
//   stage C1    subdiv = to_subdiv( h )                    [8, L]
//   stage C2    h1     = conv1( silu( LN( h ) ) )          [C_next*8, L]
//               x      = h                                 (no work)
//
// Every stage reads a full-length resident input and writes a full-length
// output, so no neighbour is ever missing and the result is identical to the
// single-graph path — every op involved is either per-row or a gather. What it
// costs is one full-length round trip per stage, which is why this runs only
// past the ceiling.
//
// The skip add is done on the host on purpose: repeat_interleave + add is a
// channel repeat, so doing it here keeps the per-block graphs to a single
// input tensor instead of threading a second one through every stage.
static bool shape_dec_level_chunked( trellis2_shape_dec_model* m,
	const trellis2_shape_dec_hparams&							 hp,
	int															 lvl,
	int															 C,
	int															 prev_C,
	int															 C_next,
	int															 L,
	bool														 want_subdiv,
	const std::vector<std::vector<int32_t>>&					 nidx,
	const std::vector<float>&									 in_hch,
	const std::vector<float>&									 in_xch,
	int64_t														 chunk,
	std::vector<float>&											 out_h1,
	std::vector<float>&											 out_x,
	std::vector<float>&											 out_subdiv,
	std::string*												 error )
{
	const size_t	  es	   = sizeof( float );
	const float		  eps	   = hp.norm_eps;
	const int		  xC	   = prev_C / 8;
	const int		  rep	   = C / xC; // repeat_interleave factor
	const std::string up_prev  = "blocks." + std::to_string( lvl - 1 ) + "." + std::to_string( hp.num_blocks[lvl - 1] );
	const std::string up_this  = "blocks." + std::to_string( lvl ) + "." + std::to_string( hp.num_blocks[lvl] );
	std::string		  missing;

	auto			  W = [&]( const std::string& n ) -> ggml_tensor* {
		 auto it = m->tensors.find( n );
		 if( it == m->tensors.end() ) {
			 if( missing.empty() )
				 missing = n;
			 return nullptr;
		 }
		 return it->second;
	};

	// Run one stage over all L voxels in blocks. `in` [Cin, L] goes resident on
	// the backend because the neighbour gather reads arbitrary rows; `build`
	// appends this stage's body and returns its [Cout, n] output.
	// `needs_nbr` must say whether `build` convolves. The neighbour leaves are
	// graph inputs, so ggml_gallocr only gives them a buffer when they actually
	// appear in the graph — creating them for a stage that does not convolve
	// (the subdiv head) and then uploading into them asserts on a null buffer.
	auto run_stage = [&]( int Cin, int Cout, bool needs_nbr, const std::vector<float>& in, std::vector<float>& out, auto build ) -> bool {
		ggml_init_params pip { ggml_tensor_overhead() * 4, nullptr, true };
		ggml_context*	 pctx = ggml_init( pip );
		if( !pctx ) {
			set_error( error, "ggml_init failed (chunked level " + std::to_string( lvl ) + ")" );
			return false;
		}
		ggml_tensor*		  res  = ggml_new_tensor_2d( pctx, GGML_TYPE_F32, Cin, L );
		ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors( pctx, m->backend );
		if( !pbuf ) {
			set_error( error, "failed to allocate the chunked level input (" + std::to_string( ( size_t )Cin * L * es >> 20 ) + " MiB, level " + std::to_string( lvl ) + ")" );
			ggml_free( pctx );
			return false;
		}
		ggml_backend_tensor_set( res, in.data(), 0, in.size() * es );

		out.resize( ( size_t )Cout * L );
		bool				 ok = true;
		std::vector<int32_t> clamped;
		std::vector<float>	 mask;

		for( int64_t v0 = 0; v0 < L && ok; v0 += chunk ) {
			const int				  n		= ( int )std::min<int64_t>( chunk, ( int64_t )L - v0 );
			const size_t			  gsize = 2048;
			ggml_init_params		  ip { ggml_tensor_overhead() * gsize + ggml_graph_overhead_custom( gsize, false ), nullptr, true };
			ggml_context*			  ctx = ggml_init( ip );
			ggml_cgraph*			  gf  = ggml_new_graph_custom( ctx, gsize, false );

			std::vector<ggml_tensor*> idx_t( 27, nullptr ), mask_t( 27, nullptr );
			if( needs_nbr ) {
				for( int k = 0; k < 27; ++k ) {
					idx_t[k]  = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, n );
					mask_t[k] = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, n ); // [1, n]
					ggml_set_input( idx_t[k] );
					ggml_set_input( mask_t[k] );
				}
			}

			ggml_tensor* o = build( ctx, res, idx_t, mask_t, n, v0 );
			if( !o || !missing.empty() ) {
				set_error( error, "missing tensor: " + ( missing.empty() ? std::string( "?" ) : missing ) + " (chunked level " + std::to_string( lvl ) + ")" );
				ggml_free( ctx );
				ok = false;
				break;
			}
			o = ggml_cont( ctx, o );
			ggml_set_output( o );
			ggml_build_forward_expand( gf, o );

			ggml_gallocr_t alloc = ggml_gallocr_new( ggml_backend_get_default_buffer_type( m->backend ) );
			if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
				set_error( error, "ggml_gallocr_alloc_graph failed (chunked level " + std::to_string( lvl ) + ", voxels from " + std::to_string( v0 ) + ")" );
				ok = false;
			} else {
				clamped.resize( ( size_t )n );
				mask.resize( ( size_t )n );
				for( int k = 0; needs_nbr && k < 27; ++k ) {
					const std::vector<int32_t>& ik = nidx[k];
					for( int i = 0; i < n; ++i ) {
						const int32_t nb	 = ik[( size_t )( v0 + i )];
						const bool	  miss	 = nb >= L;
						clamped[( size_t )i] = miss ? 0 : nb;
						mask[( size_t )i]	 = miss ? 0.0f : 1.0f;
					}
					ggml_backend_tensor_set( idx_t[k], clamped.data(), 0, ( size_t )n * sizeof( int32_t ) );
					ggml_backend_tensor_set( mask_t[k], mask.data(), 0, ( size_t )n * sizeof( float ) );
				}
				if( ggml_backend_graph_compute( m->backend, gf ) != GGML_STATUS_SUCCESS ) {
					set_error( error, "graph compute failed (chunked level " + std::to_string( lvl ) + ", voxels from " + std::to_string( v0 ) + ")" );
					ok = false;
				} else {
					ggml_backend_tensor_get( o, out.data() + ( size_t )v0 * Cout, 0, ( size_t )Cout * n * es );
				}
			}
			ggml_gallocr_free( alloc );
			ggml_free( ctx );
		}

		ggml_backend_buffer_free( pbuf );
		ggml_free( pctx );
		return ok;
	};

	// Graph fragments, in the same shapes the single-graph path builds.
	// `pre` is applied to each gathered [Ci, n] slab before the mask multiply.
	// The single-graph path applies those ops to the whole [Ci, L] tensor and
	// convolves the result; doing them here instead is what keeps every op
	// block-sized. Legal because the only ops that ever sit there are norm,
	// silu and an affine LN — all strictly per-row, so they commute with the
	// gather. Applying them to the resident tensor instead would rebuild a
	// full-length intermediate and defeat the split entirely.
	auto conv_blk = [&]( ggml_context* ctx, ggml_tensor* x, const std::string& pfx, std::vector<ggml_tensor*>& idx_t, std::vector<ggml_tensor*>& mask_t, auto pre ) -> ggml_tensor* {
		ggml_tensor* w = W( pfx + ".weight" ); // [Ci, 27, Co]
		ggml_tensor* b = W( pfx + ".bias" );
		if( !w || !b )
			return nullptr;
		const int64_t Ci = w->ne[0], Co = w->ne[2];
		ggml_tensor*  acc = nullptr;
		for( int k = 0; k < 27; ++k ) {
			ggml_tensor* wk = ggml_cont( ctx, ggml_view_3d( ctx, w, Ci, 1, Co, w->nb[1], w->nb[2], ( size_t )k * w->nb[1] ) );
			wk				= ggml_reshape_2d( ctx, wk, Ci, Co );
			ggml_tensor* g	= ggml_get_rows( ctx, x, idx_t[k] ); // [Ci, n]
			g				= pre( ctx, g );
			if( !g )
				return nullptr;
			g			   = ggml_mul( ctx, g, mask_t[k] );
			ggml_tensor* y = ggml_mul_mat( ctx, wk, g );
			acc			   = acc ? ggml_add( ctx, acc, y ) : y;
		}
		return ggml_add( ctx, acc, b );
	};
	auto no_pre = []( ggml_context*, ggml_tensor* g ) { return g; };
	auto lin_blk = [&]( ggml_context* ctx, ggml_tensor* in, const std::string& pfx ) -> ggml_tensor* {
		ggml_tensor* w = W( pfx + ".weight" );
		if( !w )
			return nullptr;
		ggml_tensor* y = ggml_mul_mat( ctx, w, in );
		ggml_tensor* b = W( pfx + ".bias" );
		if( b )
			y = ggml_add( ctx, y, b );
		return y;
	};
	auto ln_affine_blk = [&]( ggml_context* ctx, ggml_tensor* h, const std::string& pfx ) -> ggml_tensor* {
		ggml_tensor* w = W( pfx + ".weight" );
		ggml_tensor* b = W( pfx + ".bias" );
		if( !w || !b )
			return nullptr;
		ggml_tensor* y = ggml_norm( ctx, h, eps );
		y			   = ggml_mul( ctx, y, w );
		return ggml_add( ctx, y, b );
	};

	// ── stage A: the previous level's up-block tail, conv2 half ──────────────
	std::vector<float> cur;
	if( !run_stage( C, C, /*needs_nbr*/ true, in_hch, cur, [&]( ggml_context* ctx, ggml_tensor* res, std::vector<ggml_tensor*>& idx_t, std::vector<ggml_tensor*>& mask_t, int /*n*/, int64_t /*v0*/ ) -> ggml_tensor* {
			// norm2 is affine-free and silu is elementwise, so both ride behind
			// the gather (see conv_blk).
			return conv_blk( ctx, res, up_prev + ".conv2", idx_t, mask_t, [&]( ggml_context* c, ggml_tensor* g ) { return ggml_silu( c, ggml_norm( c, g, eps ) ); } );
		} ) )
		return false;

	// host: + repeat_interleave(xch). Channel repeat, so a plain strided add.
	for( int64_t v = 0; v < L; ++v ) {
		float*		 dst = cur.data() + ( size_t )v * C;
		const float* src = in_xch.data() + ( size_t )v * xC;
		for( int c = 0; c < xC; ++c )
			for( int t = 0; t < rep; ++t )
				dst[( size_t )c * rep + t] += src[c];
	}

	// ── stages B: the ConvNeXt chain ─────────────────────────────────────────
	for( int b = 0; b < hp.num_blocks[lvl]; ++b ) {
		const std::string  pfx = "blocks." + std::to_string( lvl ) + "." + std::to_string( b );
		std::vector<float> nxt;
		if( !run_stage( C, C, /*needs_nbr*/ true, cur, nxt, [&]( ggml_context* ctx, ggml_tensor* res, std::vector<ggml_tensor*>& idx_t, std::vector<ggml_tensor*>& mask_t, int n, int64_t v0 ) -> ggml_tensor* {
				ggml_tensor* h = conv_blk( ctx, res, pfx + ".conv", idx_t, mask_t, no_pre );
				if( !h )
					return nullptr;
				h = ln_affine_blk( ctx, h, pfx + ".norm" );
				h = lin_blk( ctx, h, pfx + ".mlp.0" );
				if( !h )
					return nullptr;
				h = ggml_silu( ctx, h );
				h = lin_blk( ctx, h, pfx + ".mlp.2" );
				if( !h )
					return nullptr;
				// residual: this block's own rows of the resident input
				ggml_tensor* x0 = ggml_cont( ctx, ggml_view_2d( ctx, res, C, n, res->nb[1], ( size_t )v0 * res->nb[1] ) );
				return ggml_add( ctx, h, x0 );
			} ) )
			return false;
		cur.swap( nxt );
	}

	// ── stages C: the up-block heads ─────────────────────────────────────────
	if( want_subdiv ) {
		if( !run_stage( C, 8, /*needs_nbr*/ false, cur, out_subdiv, [&]( ggml_context* ctx, ggml_tensor* res, std::vector<ggml_tensor*>&, std::vector<ggml_tensor*>&, int n, int64_t v0 ) -> ggml_tensor* {
				ggml_tensor* x0 = ggml_cont( ctx, ggml_view_2d( ctx, res, C, n, res->nb[1], ( size_t )v0 * res->nb[1] ) );
				return lin_blk( ctx, x0, up_this + ".to_subdiv" );
			} ) )
			return false;
	}
	if( !run_stage( C, C_next * 8, /*needs_nbr*/ true, cur, out_h1, [&]( ggml_context* ctx, ggml_tensor* res, std::vector<ggml_tensor*>& idx_t, std::vector<ggml_tensor*>& mask_t, int /*n*/, int64_t /*v0*/ ) -> ggml_tensor* {
			// norm1 is an affine LN and silu is elementwise — per-row either
			// way, so they ride behind the gather like norm2 above.
			return conv_blk( ctx, res, up_this + ".conv1", idx_t, mask_t, [&]( ggml_context* c, ggml_tensor* g ) {
				ggml_tensor* y = ln_affine_blk( c, g, up_this + ".norm1" );
				return y ? ggml_silu( c, y ) : nullptr;
			} );
		} ) )
		return false;

	out_x.swap( cur ); // "x" is the level's features, unchanged
	return true;
}

// Shared driver for the shape decoder. upsample_times < 0 runs the full decode
// (all levels + output layer; fills out_feats and out_coords) — the validated
// behavior. upsample_times in [1, n_levels-1] runs only that many subdivision
// levels and returns the expanded coordinate set in out_coords (out_feats
// untouched) — mirrors FlexiDualGridVaeDecoder.upsample().
static bool shape_dec_run( trellis2_shape_dec_model* m,
	const float*									 slat,
	int												 n_voxels,
	const int32_t*									 coords_in,
	int												 upsample_times,
	const std::vector<trellis2_subdiv_level>*		 guide,
	std::vector<trellis2_subdiv_level>*				 predicted_subs,
	bool											 pbr_scale,
	std::vector<float>&								 out_feats,
	std::vector<int32_t>&							 out_coords,
	trellis2_shape_dec_taps*						 taps,
	std::string*									 error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}
	// Texture decoder (pred_subdiv=False): the per-level subdivision is supplied
	// by the integrated shape decoder or standalone shape encoder (`guide`)
	// instead of predicted by a to_subdiv head.
	if( guide && ( int )guide->size() < m->hp.n_levels - 1 ) {
		set_error( error, "guide subdivisions shorter than n_levels-1" );
		return false;
	}

	const trellis2_shape_dec_hparams& hp	   = m->hp;
	const int						  n_levels = hp.n_levels;
	const float						  eps	   = hp.norm_eps;
	const size_t					  es	   = sizeof( float );

	if( upsample_times >= 0 && ( upsample_times < 1 || upsample_times > n_levels - 1 ) ) {
		set_error( error, "upsample_times out of range [1, n_levels-1]" );
		return false;
	}
	if( predicted_subs )
		predicted_subs->assign( ( size_t )std::max( 0, n_levels - 1 ), {} );

	std::string			 missing;

	// host-side level state
	std::vector<int32_t> coords( coords_in, coords_in + ( size_t )n_voxels * 3 );
	int					 L = n_voxels;
	std::vector<float>	 feats; // [L * C] voxel-major, current level features

	auto				 cap = [&]( const std::string& name, const float* data, size_t count ) {
		if( !taps )
			return;
		taps->names.push_back( name );
		taps->data.emplace_back( data, data + count );
	};
	auto cap_coords = [&]( const std::string& name ) {
		if( !taps )
			return;
		std::vector<float> c4( ( size_t )L * 4, 0.0f );
		for( int v = 0; v < L; ++v ) {
			c4[( size_t )v * 4 + 1] = ( float )coords[( size_t )v * 3];
			c4[( size_t )v * 4 + 2] = ( float )coords[( size_t )v * 3 + 1];
			c4[( size_t )v * 4 + 3] = ( float )coords[( size_t )v * 3 + 2];
		}
		cap( name, c4.data(), c4.size() );
	};

	// The previous up-block's outputs, already gathered down to the surviving
	// children (see the host_gather at the bottom of the loop): up_hch is the
	// conv1 output [C_next, L_child], up_xch the skip source [C/8, L_child].
	// Pre-gathering here — instead of reading back the full [C_next*8, L] conv
	// output and gathering in the next graph — keeps the finest level's ~8 GB
	// conv output from being duplicated in host RAM.
	std::vector<float> up_hch, up_xch, up_subdiv;
	int				   prev_C = 0, prev_L = 0;

	const bool		   t2_timing = std::getenv( "TRELLIS2_TIMING" ) != nullptr;
	double			   ms_nbr = 0, ms_graph = 0, ms_gather = 0;
	auto			   t_now	= [] { return std::chrono::steady_clock::now(); };
	auto			   ms_since = []( std::chrono::steady_clock::time_point a ) { return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - a ).count(); };

	for( int lvl = 0; lvl < n_levels; ++lvl ) {
		const int						  C		 = hp.channels[lvl];
		const bool						  has_up = lvl < n_levels - 1;
		const int						  C_next = has_up ? hp.channels[lvl + 1] : 0;

		// ── host: neighbor maps for this level's coords ─────────────────────
		std::vector<std::vector<int32_t>> nidx;
		const bool						  needs_conv = hp.num_blocks[lvl] > 0 || has_up || lvl > 0;
		if( needs_conv ) {
			auto t0 = t_now();
			build_neighbor_indices( coords, L, nidx );
			ms_nbr += ms_since( t0 );
		}

		// ── shared tail of a level with an up-block ─────────────────────────
		// Pick the surviving children from the subdivision, hand them to the
		// caller's gather, and advance the level state. Both the single-graph
		// path and the chunked one end here, and they must end here identically
		// — the child selection is what fixes the next level's voxel order, so
		// two copies of it would be a divergence nobody finds later.
		//
		// The gather itself stays with the caller rather than moving in here:
		// the graph path reads its conv output in place where the backend
		// allows it, instead of duplicating the full [C_next*8, L] (~8 GB at
		// 1024^3), and the chunked path already holds host vectors. Only the
		// index set is shared.
		std::vector<int32_t> child_coords, cidx;
		auto				 advance_up_level = [&]( auto gather ) -> bool {
			// Child slot o + 8*parent, in [0, 8L). Shape decoder: expand from
			// the predicted subdivision logits. Texture decoder: replay the
			// encoder's recorded subdivision (`guide`), which also reproduces
			// the encoder's exact input voxel order.
			child_coords.clear();
			cidx.clear();
			if( guide ) {
				const trellis2_subdiv_level& g = ( *guide )[lvl];
				child_coords				   = g.fine_coords;
				cidx						   = g.cidx;
			} else {
				child_coords.reserve( ( size_t )L * 3 );
				cidx.reserve( ( size_t )L );
				for( int v = 0; v < L; ++v ) {
					for( int o = 0; o < 8; ++o ) {
						if( up_subdiv[( size_t )v * 8 + o] > 0.0f ) {
							cidx.push_back( o + 8 * v );
							child_coords.push_back( 2 * coords[( size_t )v * 3] + ( o & 1 ) );
							child_coords.push_back( 2 * coords[( size_t )v * 3 + 1] + ( ( o >> 1 ) & 1 ) );
							child_coords.push_back( 2 * coords[( size_t )v * 3 + 2] + ( ( o >> 2 ) & 1 ) );
						}
					}
				}
			}
			if( predicted_subs ) {
				trellis2_subdiv_level& sub = ( *predicted_subs )[( size_t )lvl];
				sub.fine_coords			   = child_coords;
				sub.cidx				   = cidx;
			}
			const int L_child = ( int )cidx.size();
			if( L_child == 0 ) {
				set_error( error, "no children at level " + std::to_string( lvl ) );
				return false;
			}
			gather( cidx, L_child );
			prev_C = C;
			prev_L = L;
			coords = std::move( child_coords );
			L	   = L_child;
			return true;
		};

		// ── too many voxels for one graph? ──────────────────────────────────
		// Only the finest level splits (see shape_dec_final_level_chunked); any
		// other level would need its predecessor materialised over all L. A
		// level that is both unsplittable and past the ceiling has to fail
		// loudly — that is exactly the case that used to cost the 1024^3 mesh a
		// quarter of its faces without a word. The ceiling follows the type of
		// the weights this level's matmuls actually use.
		{
			const std::string wname	   = lvl == 0 ? std::string( "from_latent.weight" ) : ( "blocks." + std::to_string( lvl - 1 ) + "." + std::to_string( hp.num_blocks[lvl - 1] ) + ".conv2.weight" );
			auto			  wit	   = m->tensors.find( wname );
			const int64_t	  max_rows = wit == m->tensors.end() ? mul_mat_max_rows( GGML_TYPE_F32 ) : mul_mat_max_rows( wit->second->type );
			const int64_t	  block	   = std::min( chunk_rows_limit(), max_rows );

			// Per-level voxel counts. The cascade's token budget bounds the HR
			// flow's INPUT; nothing bounds the decoder's OUTPUT, and it is the
			// output that meets the mul_mat ceiling (plan D4 predicted exactly
			// this). Logging the counts is what lets the ratio between the two
			// be calibrated from real runs instead of assumed.
			if( t2_timing )
				std::fprintf( stderr, "[shape_dec] level %d: L=%d, mul_mat column cap %lld, split block %lld%s\n", lvl, L, ( long long )max_rows, ( long long )block, ( int64_t )L > max_rows ? "  <-- PAST THE CAP" : "" );

			if( ( int64_t )L > block ) {
				const bool splittable = !has_up && lvl > 0 && hp.num_blocks[lvl] == 0 && !taps;
				if( splittable ) {
					if( !shape_dec_final_level_chunked( m, hp, lvl, C, prev_C, L, nidx, up_hch, up_xch, pbr_scale, block, out_feats, error ) ) {
						return false;
					}
					out_coords = coords;
					if( t2_timing )
						std::fprintf( stderr, "[shape_dec] nbr=%.0f graph=%.0f gather=%.0f ms\n", ms_nbr, ms_graph, ms_gather );
					return true;
				}
				// An intermediate level: a ConvNeXt chain plus an up-block. Split
				// per STAGE rather than per level (shape_dec_level_chunked), so
				// the chained blocks never need a halo. Taps are excluded — they
				// want the full-length intermediates this path never forms.
				if( has_up && lvl > 0 && !taps ) {
					std::vector<float> ch_h1, ch_x;
					auto			   t_c = t_now();
					if( !shape_dec_level_chunked( m, hp, lvl, C, prev_C, C_next, L, /*want_subdiv*/ !guide, nidx, up_hch, up_xch, block, ch_h1, ch_x, up_subdiv, error ) )
						return false;
					if( t2_timing )
						ms_graph += ms_since( t_c );
					if( !advance_up_level( [&]( const std::vector<int32_t>& ci, int L_child ) {
							auto t_gh		= t_now();
							auto vec_gather = [&]( const std::vector<float>& src, int chans, std::vector<float>& out ) {
								out.resize( ( size_t )chans * L_child );
								for( int j = 0; j < L_child; ++j )
									std::memcpy( out.data() + ( size_t )j * chans, src.data() + ( size_t )ci[( size_t )j] * chans, ( size_t )chans * es );
							};
							vec_gather( ch_h1, C_next, up_hch ); // [C_next, L_child]
							vec_gather( ch_x, C / 8, up_xch );	 // [C/8,     L_child]
							if( t2_timing )
								ms_gather += ms_since( t_gh );
						} ) )
						return false;
					// Same early-out the single-graph path takes at the bottom
					// of the loop: `coords` now holds this level's child set.
					if( upsample_times >= 0 && lvl == upsample_times - 1 ) {
						out_coords = coords;
						if( t2_timing )
							std::fprintf( stderr, "[shape_dec] nbr=%.0f graph=%.0f gather=%.0f ms\n", ms_nbr, ms_graph, ms_gather );
						return true;
					}
					continue;
				}

				if( ( int64_t )L > max_rows ) {
					set_error( error,
						"level " + std::to_string( lvl ) + " has " + std::to_string( L ) + " voxels, past this backend's mul_mat column limit (" + std::to_string( max_rows ) +
							"), and cannot be split (level 0, or taps requested)" );
					return false;
				}
				// Under the ceiling: a single graph is still correct here.
			}
		}

		// ── graph: [child head from previous level] + blocks + up part A ────
		const size_t	 gsize = 65536;
		const size_t	 mem   = ggml_tensor_overhead() * gsize + ggml_graph_overhead_custom( gsize, false );
		ggml_init_params ip { mem, nullptr, true };
		ggml_context*	 ctx = ggml_init( ip );
		ggml_cgraph*	 gf	 = ggml_new_graph_custom( ctx, gsize, false );

		auto			 W = [&]( const std::string& n ) -> ggml_tensor* {
			auto it = m->tensors.find( n );
			if( it == m->tensors.end() ) {
				if( missing.empty() )
					missing = n;
				return nullptr;
			}
			return it->second;
		};
		auto lin = [&]( ggml_tensor* in, const std::string& pfx ) -> ggml_tensor* {
			ggml_tensor* y = ggml_mul_mat( ctx, W( pfx + ".weight" ), in );
			ggml_tensor* b = W( pfx + ".bias" );
			if( b )
				y = ggml_add( ctx, y, b );
			return y;
		};
		auto ln_affine = [&]( ggml_tensor* h, const std::string& pfx ) -> ggml_tensor* {
			ggml_tensor* y = ggml_norm( ctx, h, eps );
			y			   = ggml_mul( ctx, y, W( pfx + ".weight" ) );
			y			   = ggml_add( ctx, y, W( pfx + ".bias" ) );
			return y;
		};

		// Per-offset neighbor leaves, shared by every conv in this level. A
		// missing neighbor is handled without an appended zero row (the CUDA
		// CONCAT/PAD kernels abort past 65535 voxels): idx_t[k] holds the
		// neighbor row *clamped* into [0, L) and mask_t[k] is 0 there, 1 for a
		// real neighbor — so get_rows gathers a valid (harmless) row and the
		// mask multiply zeroes the missing contributions. Numerically identical
		// to gathering an explicit zero row, but every op (get_rows, broadcast
		// mul, mul_mat) stays within ggml kernels that tile the voxel dimension,
		// so the decoder runs on the GPU as well as the CPU.
		std::vector<ggml_tensor*> idx_t( 27, nullptr ), mask_t( 27, nullptr );
		if( needs_conv ) {
			for( int k = 0; k < 27; ++k ) {
				idx_t[k]  = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, L );
				mask_t[k] = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, L ); // [1, L]
				ggml_set_input( idx_t[k] );
				ggml_set_input( mask_t[k] );
			}
		}

		// submanifold conv: x [Cin, L] -> [Cout, L]
		auto conv = [&]( ggml_tensor* x, const std::string& pfx ) -> ggml_tensor* {
			ggml_tensor* w = W( pfx + ".weight" ); // ne [Ci, 27, Co]
			ggml_tensor* b = W( pfx + ".bias" );   // [Co]
			if( !w || !b )
				return x;
			const int64_t Ci = w->ne[0], Co = w->ne[2];
			ggml_tensor*  acc = nullptr;
			for( int k = 0; k < 27; ++k ) {
				ggml_tensor* wk = ggml_cont( ctx, ggml_view_3d( ctx, w, Ci, 1, Co, w->nb[1], w->nb[2], ( size_t )k * w->nb[1] ) );
				wk				= ggml_reshape_2d( ctx, wk, Ci, Co );
				ggml_tensor* g	= ggml_get_rows( ctx, x, idx_t[k] ); // [Ci, L]
				g				= ggml_mul( ctx, g, mask_t[k] );	 // zero missing (broadcast [1,L])
				ggml_tensor* y	= ggml_mul_mat( ctx, wk, g );		 // [Co, L]
				acc				= acc ? ggml_add( ctx, acc, y ) : y;
			}
			return ggml_add( ctx, acc, b );
		};

		// ConvNeXt block: x + mlp(LN(conv(x)))
		auto convnext = [&]( ggml_tensor* x, const std::string& pfx ) -> ggml_tensor* {
			ggml_tensor* h = conv( x, pfx + ".conv" );
			h			   = ln_affine( h, pfx + ".norm" );
			h			   = lin( h, pfx + ".mlp.0" );
			h			   = ggml_silu( ctx, h );
			h			   = lin( h, pfx + ".mlp.2" );
			return ggml_add( ctx, h, x );
		};

		ggml_tensor* h	  = nullptr;
		ggml_tensor *in_a = nullptr, *in_hch = nullptr, *in_xch = nullptr;

		if( lvl == 0 ) {
			// from_latent on the input slat
			in_a = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, hp.latent_channels, L );
			ggml_set_input( in_a );
			h = lin( in_a, "from_latent" );
		} else {
			// child head of the previous level's up-block. The surviving
			// children were already gathered on host (up_hch/up_xch), so we
			// receive hch [C, L] and xch [prev_C/8, L] directly — no on-device
			// get_rows over the full [C*8, prev_L]:
			//   h = conv2(silu(LN_free(hch))) + repeat_interleave(xch)
			const std::string up = "blocks." + std::to_string( lvl - 1 ) + "." + std::to_string( hp.num_blocks[lvl - 1] );
			in_hch				 = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, C, L );			// hch
			in_xch				 = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, prev_C / 8, L ); // xch
			ggml_set_input( in_hch );
			ggml_set_input( in_xch );

			const int	 r	  = C / ( prev_C / 8 ); // repeat_interleave factor
			ggml_tensor* skip = ggml_reshape_3d( ctx, in_xch, 1, prev_C / 8, L );
			skip			  = ggml_repeat( ctx, skip, ggml_new_tensor_3d( ctx, GGML_TYPE_F32, r, prev_C / 8, L ) );
			skip			  = ggml_reshape_2d( ctx, skip, C, L );

			ggml_tensor* hn = ggml_norm( ctx, in_hch, eps ); // norm2, affine-free
			hn				= ggml_silu( ctx, hn );
			hn				= conv( hn, up + ".conv2" );
			h				= ggml_add( ctx, hn, skip );
		}

		for( int b = 0; b < hp.num_blocks[lvl]; ++b ) {
			const std::string pfx = "blocks." + std::to_string( lvl ) + "." + std::to_string( b );
			h					  = convnext( h, pfx );
		}

		std::vector<std::pair<std::string, ggml_tensor*>> outs;
		if( has_up ) {
			const std::string up = "blocks." + std::to_string( lvl ) + "." + std::to_string( hp.num_blocks[lvl] );
			if( !guide ) {										   // shape decoder: predict which children exist
				ggml_tensor* subdiv = lin( h, up + ".to_subdiv" ); // [8, L]
				outs.emplace_back( "subdiv", ggml_cont( ctx, subdiv ) );
			}
			ggml_tensor* hn = ln_affine( h, up + ".norm1" );
			hn				= ggml_silu( ctx, hn );
			ggml_tensor* h1 = conv( hn, up + ".conv1" ); // [C_next*8, L]
			outs.emplace_back( "h1", ggml_cont( ctx, h1 ) );
			outs.emplace_back( "x", ggml_cont( ctx, h ) );
		} else {
			// final: affine-free LN (eps 1e-5) + output projection
			ggml_tensor* hn = ggml_norm( ctx, h, 1e-5f );
			ggml_tensor* o	= lin( hn, "output_layer" ); // [7, L]
			outs.emplace_back( "out", ggml_cont( ctx, o ) );
		}
		if( taps )
			outs.emplace_back( "pre_up", ggml_cont( ctx, h ) );

		for( auto& o : outs ) {
			ggml_set_output( o.second );
			ggml_build_forward_expand( gf, o.second );
		}

		if( !missing.empty() ) {
			set_error( error, "missing tensor: " + missing + " (level " + std::to_string( lvl ) + ")" );
			ggml_free( ctx );
			return false;
		}

		ggml_gallocr_t alloc = ggml_gallocr_new( ggml_backend_get_default_buffer_type( m->backend ) );
		if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
			set_error( error, "ggml_gallocr_alloc_graph failed (level " + std::to_string( lvl ) + ")" );
			ggml_gallocr_free( alloc );
			ggml_free( ctx );
			return false;
		}

		// upload inputs: clamp the missing-neighbor sentinel (L) into range and
		// build its 0/1 mask (see the conv leaves above).
		if( needs_conv ) {
			std::vector<int32_t> clamped( ( size_t )L );
			std::vector<float>	 mask( ( size_t )L );
			for( int k = 0; k < 27; ++k ) {
				const std::vector<int32_t>& ik = nidx[k];
				for( int v = 0; v < L; ++v ) {
					const bool miss		 = ik[( size_t )v] >= L;
					clamped[( size_t )v] = miss ? 0 : ik[( size_t )v];
					mask[( size_t )v]	 = miss ? 0.0f : 1.0f;
				}
				ggml_backend_tensor_set( idx_t[k], clamped.data(), 0, ( size_t )L * sizeof( int32_t ) );
				ggml_backend_tensor_set( mask_t[k], mask.data(), 0, ( size_t )L * sizeof( float ) );
			}
		}
		if( lvl == 0 ) {
			ggml_backend_tensor_set( in_a, slat, 0, ( size_t )hp.latent_channels * L * es );
		} else {
			// pre-gathered by the previous level (host_gather below)
			ggml_backend_tensor_set( in_hch, up_hch.data(), 0, up_hch.size() * es );
			ggml_backend_tensor_set( in_xch, up_xch.data(), 0, up_xch.size() * es );
		}

		auto			  t_g = t_now();
		const ggml_status st  = ggml_backend_graph_compute( m->backend, gf );
		if( t2_timing )
			ms_graph += ms_since( t_g );
		if( st != GGML_STATUS_SUCCESS ) {
			set_error( error, "graph compute failed (level " + std::to_string( lvl ) + ")" );
			ggml_gallocr_free( alloc );
			ggml_free( ctx );
			return false;
		}

		// taps + read-back
		if( taps ) {
			cap_coords( "lvl" + std::to_string( lvl ) + ".in_coords" );
		}
		for( auto& o : outs ) {
			if( o.first == "pre_up" ) {
				std::vector<float> buf( ( size_t )ggml_nelements( o.second ) );
				ggml_backend_tensor_get( o.second, buf.data(), 0, buf.size() * es );
				cap( "lvl" + std::to_string( lvl ) + ".pre_up", buf.data(), buf.size() );
			}
		}

		if( has_up ) {
			ggml_tensor *h1_o = nullptr, *x_o = nullptr, *subdiv_o = nullptr;
			for( auto& o : outs ) {
				if( o.first == "subdiv" )
					subdiv_o = o.second;
				if( o.first == "h1" )
					h1_o = o.second;
				if( o.first == "x" )
					x_o = o.second;
			}
			if( !guide ) { // shape decoder: read the predicted subdivision logits
				up_subdiv.resize( ( size_t )8 * L );
				ggml_backend_tensor_get( subdiv_o, up_subdiv.data(), 0, up_subdiv.size() * es );
				if( taps )
					cap( "lvl" + std::to_string( lvl ) + ".subdiv", up_subdiv.data(), up_subdiv.size() );
			}

			// Gather the surviving children of h1 [C_next*8, L] (viewed as
			// [C_next, 8L]) and x [C, L] (viewed as [C/8, 8L]) — byte-identical
			// to get_rows(cidx) on the reshaped tensors. On the CPU backend the
			// conv output is already in host memory, so we read it in place
			// instead of duplicating the full [C_next*8, L] (~8 GB at 1024^3).
			if( !advance_up_level( [&]( const std::vector<int32_t>& ci, int L_child ) {
					auto host_gather = [&]( ggml_tensor* t, int chans, std::vector<float>& out ) {
						out.resize( ( size_t )chans * L_child );
						if( t->buffer && ggml_backend_buffer_is_host( t->buffer ) ) {
							const float* d = ( const float* )t->data;
							for( int j = 0; j < L_child; ++j )
								std::memcpy( out.data() + ( size_t )j * chans, d + ( size_t )ci[( size_t )j] * chans, ( size_t )chans * es );
						} else {
							std::vector<float> full( ( size_t )ggml_nelements( t ) );
							ggml_backend_tensor_get( t, full.data(), 0, full.size() * es );
							for( int j = 0; j < L_child; ++j )
								std::memcpy( out.data() + ( size_t )j * chans, full.data() + ( size_t )ci[( size_t )j] * chans, ( size_t )chans * es );
						}
					};
					auto t_gh = t_now();
					host_gather( h1_o, C_next, up_hch ); // [C_next, L_child]
					host_gather( x_o, C / 8, up_xch );	 // [C/8,     L_child]
					if( t2_timing )
						ms_gather += ms_since( t_gh );
				} ) ) {
				ggml_gallocr_free( alloc );
				ggml_free( ctx );
				return false;
			}
		} else {
			out_feats.resize( ( size_t )hp.out_channels * L );
			for( auto& o : outs ) {
				if( o.first == "out" ) {
					ggml_backend_tensor_get( o.second, out_feats.data(), 0, out_feats.size() * es );
				}
			}
			if( pbr_scale ) // tex decoder: map to [0,1] like the reference *0.5+0.5
				for( float& f : out_feats )
					f = f * 0.5f + 0.5f;
			out_coords = coords;
			if( taps ) {
				cap( pbr_scale ? "pbr" : "out7", out_feats.data(), out_feats.size() );
				cap_coords( "out_coords" );
			}
		}

		ggml_gallocr_free( alloc );
		ggml_free( ctx );

		// upsample-to-level-N: at this point `coords` holds the child set this
		// level's up-block produced (has_up is always true for lvl < n_levels-1,
		// which upsample_times-1 always is), so return it and skip the rest.
		if( upsample_times >= 0 && lvl == upsample_times - 1 ) {
			out_coords = coords;
			if( t2_timing )
				std::fprintf( stderr, "[shape_dec] nbr=%.0f graph=%.0f gather=%.0f ms\n", ms_nbr, ms_graph, ms_gather );
			return true;
		}
	}

	if( t2_timing )
		std::fprintf( stderr, "[shape_dec] nbr=%.0f graph=%.0f gather=%.0f ms\n", ms_nbr, ms_graph, ms_gather );
	return true;
}

bool trellis2_shape_dec_decode( trellis2_shape_dec_model* m,
	const float*										  slat,
	int													  n_voxels,
	const int32_t*										  coords_in,
	std::vector<float>&									  out_feats,
	std::vector<int32_t>&								  out_coords,
	trellis2_shape_dec_taps*							  taps,
	std::string*										  error )
{
	return shape_dec_run( m,
		slat,
		n_voxels,
		coords_in,
		/*upsample_times*/ -1,
		/*guide*/ nullptr,
		/*predicted_subs*/ nullptr,
		/*pbr_scale*/ false,
		out_feats,
		out_coords,
		taps,
		error );
}

bool trellis2_shape_dec_decode_with_subs( trellis2_shape_dec_model* m,
	const float*													slat,
	int																n_voxels,
	const int32_t*													coords_in,
	std::vector<float>&												out_feats,
	std::vector<int32_t>&											out_coords,
	std::vector<trellis2_subdiv_level>&								out_subs,
	trellis2_shape_dec_taps*										taps,
	std::string*													error )
{
	return shape_dec_run( m,
		slat,
		n_voxels,
		coords_in,
		/*upsample_times*/ -1,
		/*guide*/ nullptr,
		&out_subs,
		/*pbr_scale*/ false,
		out_feats,
		out_coords,
		taps,
		error );
}

bool trellis2_shape_dec_upsample( trellis2_shape_dec_model* m, const float* slat, int n_voxels, const int32_t* coords, int upsample_times, std::vector<int32_t>& out_coords, std::string* error )
{
	std::vector<float> unused;
	return shape_dec_run( m,
		slat,
		n_voxels,
		coords,
		upsample_times,
		/*guide*/ nullptr,
		/*predicted_subs*/ nullptr,
		/*pbr_scale*/ false,
		unused,
		out_coords,
		/*taps*/ nullptr,
		error );
}

// Texture decoder: the shape-decoder driver with the supplied shape subdivision
// replayed (guide) and the PBR [0,1] output scale.
bool trellis2_tex_dec_decode( trellis2_shape_dec_model* m,
	const float*										slat,
	int													n_voxels,
	const int32_t*										coords,
	const std::vector<trellis2_subdiv_level>&			subs,
	std::vector<float>&									out_feats,
	std::vector<int32_t>&								out_coords,
	std::string*										error )
{
	return shape_dec_run( m, slat, n_voxels, coords, /*upsample_times*/ -1, &subs, /*predicted_subs*/ nullptr, /*pbr_scale*/ true, out_feats, out_coords, /*taps*/ nullptr, error );
}

/*****************************************************************************
** Shape-SLAT VAE encoder (FlexiDualGridVaeEncoder) — the mirror of the shape
** decoder. The 6-channel dual grid at resolution R is downsampled 16x through
** SparseSpatial2Channel (S2C) blocks into the 32-channel shape SLAT at R/16.
** Unlike the decoder, an S2C step needs no learned decision — the parent set is
** determined by the coordinates alone — so a whole level (ConvNeXt blocks + the
** down-block's fine conv1, the S2C gather, and the coarse conv2) runs as one
** graph. The per-level subdivision (which fine child maps to which coarse
** parent) is recorded for the texture decoder to replay.
*****************************************************************************/

struct trellis2_shape_enc_model {
	gguf_context*								  gguf = nullptr;
	ggml_context*								  ctx  = nullptr;
	trellis2_shape_enc_hparams					  hp;
	bool										  has_data = false;

	ggml_backend_t								  backend	  = nullptr;
	ggml_backend_buffer_t						  weights_buf = nullptr;
	std::string									  backend_name;

	std::unordered_map<std::string, ggml_tensor*> tensors;
};

trellis2_shape_enc_model* trellis2_shape_enc_load( const std::string& path, bool load_tensors, std::string* error, const char* device )
{
	auto*			 m = new trellis2_shape_enc_model();

	gguf_init_params params;
	params.no_alloc = true;
	params.ctx		= &m->ctx;
	m->gguf			= gguf_init_from_file( path.c_str(), params );
	if( !m->gguf ) {
		set_error( error, "gguf_init_from_file failed (not a GGUF file?): " + path );
		delete m;
		return nullptr;
	}

	const char* arch = kv_str( m->gguf, "general.architecture", "" );
	if( std::strcmp( arch, "trellis2-shape-enc" ) != 0 ) {
		set_error( error, std::string( "unexpected architecture '" ) + arch + "' (expected 'trellis2-shape-enc')" );
		trellis2_shape_enc_free( m );
		return nullptr;
	}

	trellis2_shape_enc_hparams& hp = m->hp;
	const char*					P  = "trellis2.shape_enc.";
	auto						K  = [&]( const std::string& s ) { return std::string( P ) + s; };
	hp.in_channels				   = ( int32_t )kv_u32( m->gguf, K( "in_channels" ).c_str(), 6 );
	hp.latent_channels			   = ( int32_t )kv_u32( m->gguf, K( "latent_channels" ).c_str(), 0 );
	hp.n_levels					   = ( int32_t )kv_u32( m->gguf, K( "n_levels" ).c_str(), 0 );
	hp.norm_eps					   = kv_f32( m->gguf, K( "norm_eps" ).c_str(), 1e-6f );
	hp.file_type				   = ( int32_t )kv_u32( m->gguf, "general.file_type", 0 );
	for( int i = 0; i < hp.n_levels && i < 8; ++i ) {
		hp.channels[i]	 = ( int32_t )kv_u32( m->gguf, K( "channels." + std::to_string( i ) ).c_str(), 0 );
		hp.num_blocks[i] = ( int32_t )kv_u32( m->gguf, K( "num_blocks." + std::to_string( i ) ).c_str(), 0 );
	}

	for( ggml_tensor* t = ggml_get_first_tensor( m->ctx ); t != nullptr; t = ggml_get_next_tensor( m->ctx, t ) ) {
		m->tensors[t->name] = t;
	}

	if( load_tensors ) {
		m->backend	   = init_best_backend( m->backend_name, device );
		m->weights_buf = ggml_backend_alloc_ctx_tensors( m->ctx, m->backend );
		if( !m->weights_buf ) {
			set_error( error, "failed to allocate weights on backend " + m->backend_name );
			trellis2_shape_enc_free( m );
			return nullptr;
		}
		std::ifstream fin( path, std::ios::binary );
		if( !fin ) {
			set_error( error, "cannot reopen file for weight data: " + path );
			trellis2_shape_enc_free( m );
			return nullptr;
		}
		const size_t		 data_off = gguf_get_data_offset( m->gguf );
		const int64_t		 nt		  = gguf_get_n_tensors( m->gguf );
		std::vector<uint8_t> buf;
		for( int64_t i = 0; i < nt; ++i ) {
			const char*	 name = gguf_get_tensor_name( m->gguf, i );
			ggml_tensor* t	  = m->tensors[name];
			const size_t nb	  = ggml_nbytes( t );
			const size_t off  = data_off + gguf_get_tensor_offset( m->gguf, i );
			buf.resize( nb );
			fin.seekg( ( std::streamoff )off, std::ios::beg );
			if( !fin.read( reinterpret_cast<char*>( buf.data() ), ( std::streamsize )nb ) ) {
				set_error( error, std::string( "failed reading weight '" ) + name + "' from file" );
				trellis2_shape_enc_free( m );
				return nullptr;
			}
			ggml_backend_tensor_set( t, buf.data(), 0, nb );
		}
		m->has_data = true;
	}
	return m;
}

void trellis2_shape_enc_free( trellis2_shape_enc_model* m )
{
	if( !m )
		return;
	if( m->weights_buf )
		ggml_backend_buffer_free( m->weights_buf );
	if( m->backend )
		ggml_backend_free( m->backend );
	if( m->gguf )
		gguf_free( m->gguf );
	if( m->ctx )
		ggml_free( m->ctx );
	delete m;
}

const char* trellis2_shape_enc_backend_name( const trellis2_shape_enc_model* m )
{
	return ( m && !m->backend_name.empty() ) ? m->backend_name.c_str() : "none";
}

const trellis2_shape_enc_hparams& trellis2_shape_enc_hparams_of( const trellis2_shape_enc_model* m )
{
	return m->hp;
}

// The shape encoder's first level, evaluated in voxel blocks.
//
// It is the only encoder level that can grow large — Lc falls ~4x per level —
// and it is also the only one with no ConvNeXt blocks, so it needs no per-stage
// ping-pong the way shape_dec_level_chunked does. What it does need is two
// voxel spaces: the convs and the child gather live in fine space (L), the
// down-sampled result in coarse space (Lc).
//
// Why blocks at all: a GPU dispatch is sized in WORKITEMS in 32 bits, and this
// level launches several kernels whose grid scales with the voxel count — the
// skip reduction over C_out*Lc rows, and the S2C gather over Lc*8 index rows.
// Both cross 2^32 somewhere around Lc = 2^20..2^21, which a 1536 cascade
// reaches on a dense object. Chasing them one kernel at a time does not
// converge; bounding every launch at once does.
// See docs/progress/1536-cascade_backend-limits.md.
//
//   P0  h    = input_layer(in)                       [C_in, L]     per-row
//   P1  h1   = conv1(silu(LN(h)))                    [C_out/8, L]  27 fine nbrs
//   P2  h1c  = s2c(h1)                               [C_out, Lc]   8 children
//   P3  skip = mean_gsz(s2c(h))                      [C_out, Lc]   8 children
//   P4  out  = conv2(silu(LN_free(h1c))) + skip      [C_out, Lc]   27 coarse nbrs
//
// Every pass reads a full-length resident input and gathers from it, so no
// neighbour or child is ever absent and the result matches the single-graph
// path exactly — every op is per-row or a gather.
static bool shape_enc_level_chunked( trellis2_shape_enc_model* m,
	const trellis2_shape_enc_hparams&							 hp,
	int															 lvl,
	int															 C_in,
	int															 C_out,
	int															 L,
	int															 Lc,
	const std::vector<float>&									 in_shift,
	const std::vector<std::vector<int32_t>>&					 nfine,
	const std::vector<std::vector<int32_t>>&					 ncoarse,
	const std::vector<int32_t>&									 childidx,
	int64_t														 chunk,
	std::vector<float>&											 out_feats,
	std::string*												 error )
{
	const size_t	  es	= sizeof( float );
	const float		  eps	= hp.norm_eps;
	const int		  in_dim = hp.in_channels;
	const int		  h1C	= C_out / 8;
	const int		  gsz	= ( C_in * 8 ) / C_out;
	const std::string down	= "blocks." + std::to_string( lvl ) + "." + std::to_string( hp.num_blocks[lvl] );
	std::string		  missing;

	auto			  W = [&]( const std::string& n ) -> ggml_tensor* {
		 auto it = m->tensors.find( n );
		 if( it == m->tensors.end() ) {
			 if( missing.empty() )
				 missing = n;
			 return nullptr;
		 }
		 return it->second;
	};

	// `build` appends one block's body and returns its [Cout, n] output;
	// `upload` fills whatever per-block inputs `build` created, after the graph
	// is allocated. They are two callbacks rather than one because ggml_gallocr
	// only gives an input tensor a buffer once it is part of the graph.
	auto run_stage = [&]( int Cin, int Lin, int Cout, int Lout, const std::vector<float>& in, std::vector<float>& out, auto build, auto upload ) -> bool {
		ggml_init_params pip { ggml_tensor_overhead() * 4, nullptr, true };
		ggml_context*	 pctx = ggml_init( pip );
		if( !pctx ) {
			set_error( error, "ggml_init failed (chunked shape_enc level 0)" );
			return false;
		}
		ggml_tensor*		  res  = ggml_new_tensor_2d( pctx, GGML_TYPE_F32, Cin, Lin );
		ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors( pctx, m->backend );
		if( !pbuf ) {
			set_error( error, "failed to allocate the chunked shape_enc input (" + std::to_string( ( size_t )Cin * Lin * es >> 20 ) + " MiB)" );
			ggml_free( pctx );
			return false;
		}
		ggml_backend_tensor_set( res, in.data(), 0, in.size() * es );

		out.resize( ( size_t )Cout * Lout );
		bool ok = true;
		for( int64_t v0 = 0; v0 < Lout && ok; v0 += chunk ) {
			const int		 n		= ( int )std::min<int64_t>( chunk, ( int64_t )Lout - v0 );
			const size_t	 gsize	= 2048;
			ggml_init_params ip { ggml_tensor_overhead() * gsize + ggml_graph_overhead_custom( gsize, false ), nullptr, true };
			ggml_context*	 ctx = ggml_init( ip );
			ggml_cgraph*	 gf	 = ggml_new_graph_custom( ctx, gsize, false );

			ggml_tensor*	 o = build( ctx, res, n, v0 );
			if( !o || !missing.empty() ) {
				set_error( error, "missing tensor: " + ( missing.empty() ? std::string( "?" ) : missing ) + " (chunked shape_enc level 0)" );
				ggml_free( ctx );
				return false;
			}
			o = ggml_cont( ctx, o );
			ggml_set_output( o );
			ggml_build_forward_expand( gf, o );

			ggml_gallocr_t alloc = ggml_gallocr_new( ggml_backend_get_default_buffer_type( m->backend ) );
			if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
				set_error( error, "ggml_gallocr_alloc_graph failed (chunked shape_enc level 0, voxels from " + std::to_string( v0 ) + ")" );
				ok = false;
			} else {
				upload( n, v0 );
				if( ggml_backend_graph_compute( m->backend, gf ) != GGML_STATUS_SUCCESS ) {
					set_error( error, "graph compute failed (chunked shape_enc level 0, voxels from " + std::to_string( v0 ) + ")" );
					ok = false;
				} else {
					ggml_backend_tensor_get( o, out.data() + ( size_t )v0 * Cout, 0, ( size_t )Cout * n * es );
				}
			}
			ggml_gallocr_free( alloc );
			ggml_free( ctx );
		}
		ggml_backend_buffer_free( pbuf );
		ggml_free( pctx );
		return ok;
	};

	// Per-block leaves, declared here so `build` can create them and `upload`
	// can fill them.
	std::vector<ggml_tensor*> idx_t( 27, nullptr ), mask_t( 27, nullptr );
	ggml_tensor *			  cidx_t = nullptr, *cmask_t = nullptr, *skip_t = nullptr;

	auto					  make_nbr_leaves = [&]( ggml_context* ctx, int n ) {
		 for( int k = 0; k < 27; ++k ) {
			 idx_t[k]  = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, n );
			 mask_t[k] = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, n );
			 ggml_set_input( idx_t[k] );
			 ggml_set_input( mask_t[k] );
		 }
	};
	auto upload_nbr_leaves = [&]( const std::vector<std::vector<int32_t>>& nb, int Ln, int n, int64_t v0 ) {
		std::vector<int32_t> cl( ( size_t )n );
		std::vector<float>	 mk( ( size_t )n );
		for( int k = 0; k < 27; ++k ) {
			for( int i = 0; i < n; ++i ) {
				const int32_t r	 = nb[k][( size_t )( v0 + i )];
				const bool	  ms = r >= Ln;
				cl[( size_t )i]	 = ms ? 0 : r;
				mk[( size_t )i]	 = ms ? 0.0f : 1.0f;
			}
			ggml_backend_tensor_set( idx_t[k], cl.data(), 0, ( size_t )n * sizeof( int32_t ) );
			ggml_backend_tensor_set( mask_t[k], mk.data(), 0, ( size_t )n * sizeof( float ) );
		}
	};
	// Submanifold conv over a block, with the per-row ops riding behind the
	// gather so every launch stays block-sized (same argument as the decoder).
	auto conv_blk = [&]( ggml_context* ctx, ggml_tensor* x, const std::string& pfx, auto pre ) -> ggml_tensor* {
		ggml_tensor* w = W( pfx + ".weight" );
		ggml_tensor* b = W( pfx + ".bias" );
		if( !w || !b )
			return nullptr;
		const int64_t Ci = w->ne[0], Co = w->ne[2];
		ggml_tensor*  acc = nullptr;
		for( int k = 0; k < 27; ++k ) {
			ggml_tensor* wk = ggml_cont( ctx, ggml_view_3d( ctx, w, Ci, 1, Co, w->nb[1], w->nb[2], ( size_t )k * w->nb[1] ) );
			wk				= ggml_reshape_2d( ctx, wk, Ci, Co );
			ggml_tensor* g	= ggml_get_rows( ctx, x, idx_t[k] );
			g				= pre( ctx, g );
			if( !g )
				return nullptr;
			g			   = ggml_mul( ctx, g, mask_t[k] );
			ggml_tensor* y = ggml_mul_mat( ctx, wk, g );
			acc			   = acc ? ggml_add( ctx, acc, y ) : y;
		}
		return ggml_add( ctx, acc, b );
	};
	// S2C: gather each coarse voxel's 8 children and stack them channel-wise.
	auto no_pre	  = []( ggml_context*, ggml_tensor* g ) { return g; };
	auto lin_blk  = [&]( ggml_context* ctx, ggml_tensor* in, const std::string& pfx ) -> ggml_tensor* {
		 ggml_tensor* w = W( pfx + ".weight" );
		 if( !w )
			 return nullptr;
		 ggml_tensor* y = ggml_mul_mat( ctx, w, in );
		 ggml_tensor* b = W( pfx + ".bias" );
		 return b ? ggml_add( ctx, y, b ) : y;
	};
	auto ln_affine_blk = [&]( ggml_context* ctx, ggml_tensor* h, const std::string& pfx ) -> ggml_tensor* {
		ggml_tensor* w = W( pfx + ".weight" );
		ggml_tensor* b = W( pfx + ".bias" );
		if( !w || !b )
			return nullptr;
		ggml_tensor* y = ggml_norm( ctx, h, eps );
		y			   = ggml_mul( ctx, y, w );
		return ggml_add( ctx, y, b );
	};
	auto s2c_blk = [&]( ggml_context* ctx, ggml_tensor* src, int ch, int n ) -> ggml_tensor* {
		ggml_tensor* g = ggml_get_rows( ctx, src, cidx_t ); // [ch, n*8]
		g			   = ggml_mul( ctx, g, cmask_t );
		return ggml_reshape_2d( ctx, ggml_cont( ctx, g ), ( int64_t )ch * 8, n );
	};
	auto make_child_leaves = [&]( ggml_context* ctx, int n ) {
		cidx_t	= ggml_new_tensor_1d( ctx, GGML_TYPE_I32, ( int64_t )n * 8 );
		cmask_t = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, ( int64_t )n * 8 );
		ggml_set_input( cidx_t );
		ggml_set_input( cmask_t );
	};
	auto upload_child_leaves = [&]( int n, int64_t v0 ) {
		std::vector<int32_t> cl( ( size_t )n * 8 );
		std::vector<float>	 mk( ( size_t )n * 8 );
		for( size_t i = 0; i < cl.size(); ++i ) {
			const int32_t c = childidx[( size_t )v0 * 8 + i];
			cl[i]			= c < 0 ? 0 : c;
			mk[i]			= c < 0 ? 0.0f : 1.0f;
		}
		ggml_backend_tensor_set( cidx_t, cl.data(), 0, cl.size() * sizeof( int32_t ) );
		ggml_backend_tensor_set( cmask_t, mk.data(), 0, mk.size() * sizeof( float ) );
	};

	// ── P0: input_layer, per-row. Level 0 only — deeper levels take the
	//        previous level's features unchanged. ──────────────────────────
	std::vector<float> h;
	if( lvl == 0 ) {
		if( !run_stage(
				in_dim, L, C_in, L, in_shift, h,
				[&]( ggml_context* ctx, ggml_tensor* res, int n, int64_t v0 ) -> ggml_tensor* {
					ggml_tensor* x = ggml_cont( ctx, ggml_view_2d( ctx, res, in_dim, n, res->nb[1], ( size_t )v0 * res->nb[1] ) );
					ggml_tensor* w = W( "input_layer.weight" );
					if( !w )
						return nullptr;
					ggml_tensor* y = ggml_mul_mat( ctx, w, x );
					ggml_tensor* b = W( "input_layer.bias" );
					return b ? ggml_add( ctx, y, b ) : y;
				},
				[]( int, int64_t ) {} ) )
			return false;
	} else {
		h = in_shift; // already [C_in, L]
	}

	// ── PB: the ConvNeXt chain, one stage per block ─────────────────────────
	// Same reason as the decoder: block b+1 reads the NEIGHBOURS of block b's
	// output, so a per-block graph cannot see them. Each block therefore reads
	// a full-length resident input and writes a full-length output.
	for( int b = 0; b < hp.num_blocks[lvl]; ++b ) {
		const std::string  pfx = "blocks." + std::to_string( lvl ) + "." + std::to_string( b );
		std::vector<float> nxt;
		if( !run_stage(
				C_in, L, C_in, L, h, nxt,
				[&]( ggml_context* ctx, ggml_tensor* res, int n, int64_t v0 ) -> ggml_tensor* {
					make_nbr_leaves( ctx, n );
					ggml_tensor* y = conv_blk( ctx, res, pfx + ".conv", no_pre );
					if( !y )
						return nullptr;
					y = ln_affine_blk( ctx, y, pfx + ".norm" );
					y = lin_blk( ctx, y, pfx + ".mlp.0" );
					if( !y )
						return nullptr;
					y = ggml_silu( ctx, y );
					y = lin_blk( ctx, y, pfx + ".mlp.2" );
					if( !y )
						return nullptr;
					// residual: this block's own rows of the resident input
					ggml_tensor* x0 = ggml_cont( ctx, ggml_view_2d( ctx, res, C_in, n, res->nb[1], ( size_t )v0 * res->nb[1] ) );
					return ggml_add( ctx, y, x0 );
				},
				[&]( int n, int64_t v0 ) { upload_nbr_leaves( nfine, L, n, v0 ); } ) )
			return false;
		h.swap( nxt );
	}

	// ── P1: conv1 over the fine voxels ───────────────────────────────────────
	std::vector<float> h1;
	if( !run_stage(
			C_in, L, h1C, L, h, h1,
			[&]( ggml_context* ctx, ggml_tensor* res, int n, int64_t ) -> ggml_tensor* {
				make_nbr_leaves( ctx, n );
				// norm1 is an affine LN and silu is elementwise — per-row, so
				// both commute with the gather.
				return conv_blk( ctx, res, down + ".conv1", [&]( ggml_context* c, ggml_tensor* g ) -> ggml_tensor* {
					ggml_tensor* nw = W( down + ".norm1.weight" );
					ggml_tensor* nb = W( down + ".norm1.bias" );
					if( !nw || !nb )
						return nullptr;
					ggml_tensor* y = ggml_norm( c, g, eps );
					y			   = ggml_mul( c, y, nw );
					y			   = ggml_add( c, y, nb );
					return ggml_silu( c, y );
				} );
			},
			[&]( int n, int64_t v0 ) { upload_nbr_leaves( nfine, L, n, v0 ); } ) )
		return false;

	// ── P2: S2C of conv1's output -> coarse space ────────────────────────────
	std::vector<float> h1c;
	if( !run_stage(
			h1C, L, C_out, Lc, h1, h1c,
			[&]( ggml_context* ctx, ggml_tensor* res, int n, int64_t ) -> ggml_tensor* {
				make_child_leaves( ctx, n );
				return s2c_blk( ctx, res, h1C, n );
			},
			[&]( int n, int64_t v0 ) { upload_child_leaves( n, v0 ); } ) )
		return false;

	// ── P3: the down-sample skip, mean over each gsz-wide channel group ──────
	std::vector<float> skip;
	if( !run_stage(
			C_in, L, C_out, Lc, h, skip,
			[&]( ggml_context* ctx, ggml_tensor* res, int n, int64_t ) -> ggml_tensor* {
				make_child_leaves( ctx, n );
				ggml_tensor* xc = s2c_blk( ctx, res, C_in, n ); // [C_in*8, n]
				ggml_tensor* g3 = ggml_reshape_3d( ctx, xc, gsz, C_out, n );
				ggml_tensor* acc = nullptr;
				for( int j = 0; j < gsz; ++j ) {
					ggml_tensor* s = ggml_cont( ctx, ggml_view_3d( ctx, g3, 1, C_out, n, g3->nb[1], g3->nb[2], ( size_t )j * g3->nb[0] ) );
					acc			   = acc ? ggml_add( ctx, acc, s ) : s;
				}
				return ggml_scale( ctx, ggml_reshape_2d( ctx, acc, C_out, n ), 1.0f / ( float )gsz );
			},
			[&]( int n, int64_t v0 ) { upload_child_leaves( n, v0 ); } ) )
		return false;

	// ── P4: conv2 over the coarse voxels, plus the skip ──────────────────────
	if( !run_stage(
			C_out, Lc, C_out, Lc, h1c, out_feats,
			[&]( ggml_context* ctx, ggml_tensor* res, int n, int64_t ) -> ggml_tensor* {
				make_nbr_leaves( ctx, n );
				skip_t = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, C_out, n );
				ggml_set_input( skip_t );
				// norm2 is affine-free; silu elementwise. Both per-row.
				ggml_tensor* h2 = conv_blk( ctx, res, down + ".conv2", [&]( ggml_context* c, ggml_tensor* g ) { return ggml_silu( c, ggml_norm( c, g, eps ) ); } );
				return h2 ? ggml_add( ctx, h2, skip_t ) : nullptr;
			},
			[&]( int n, int64_t v0 ) {
				upload_nbr_leaves( ncoarse, Lc, n, v0 );
				ggml_backend_tensor_set( skip_t, skip.data() + ( size_t )v0 * C_out, 0, ( size_t )C_out * n * es );
			} ) )
		return false;

	return true;
}

bool trellis2_shape_enc_encode( trellis2_shape_enc_model* m,
	const float*										  in6,
	int													  n_voxels,
	const int32_t*										  coords_in,
	std::vector<float>&									  out_slat,
	std::vector<int32_t>&								  out_coords,
	std::vector<trellis2_subdiv_level>&					  out_subs,
	trellis2_shape_dec_taps*							  taps,
	std::string*										  error )
{
	if( !m ) {
		set_error( error, "null model" );
		return false;
	}
	if( !m->has_data ) {
		set_error( error, "model loaded metadata-only; reload with load_tensors=true" );
		return false;
	}

	const trellis2_shape_enc_hparams& hp		= m->hp;
	const int						  n_levels	= hp.n_levels;
	const float						  eps		= hp.norm_eps;
	const size_t					  es		= sizeof( float );
	const bool						  t2_timing = std::getenv( "TRELLIS2_TIMING" ) != nullptr;
	std::string						  missing;

	// host-side level state
	std::vector<int32_t>			  coords( coords_in, coords_in + ( size_t )n_voxels * 3 );
	int								  L = n_voxels;
	std::vector<float>				  feats; // [L * C] voxel-major, current level features

	// input to the network: (dual-vertex offset, intersected) - 0.5
	std::vector<float>				  in_shift( ( size_t )L * hp.in_channels );
	for( size_t i = 0; i < in_shift.size(); ++i )
		in_shift[i] = in6[i] - 0.5f;

	out_subs.assign( std::max( 0, n_levels - 1 ), {} );

	auto cap = [&]( const std::string& name, const float* d, size_t n ) {
		if( !taps )
			return;
		taps->names.push_back( name );
		taps->data.emplace_back( d, d + n );
	};

	for( int lvl = 0; lvl < n_levels; ++lvl ) {
		const int						  C_in	   = hp.channels[lvl];
		const bool						  has_down = lvl < n_levels - 1;
		const int						  C_out	   = has_down ? hp.channels[lvl + 1] : 0;

		// ── host: fine neighbor maps (ConvNeXt + down conv1) ────────────────
		std::vector<std::vector<int32_t>> nfine;
		build_neighbor_indices( coords, L, nfine );

		// ── host: S2C child map + coarse coords + coarse neighbor maps ──────
		// coarse parent = fine // 2, subidx = bit-packed (x&1,y&1,z&1). Coarse
		// coords are lexicographically sorted to match the reference's unique().
		std::vector<int32_t>			  coarse_coords; // [Lc*3]
		std::vector<int32_t>			  childidx;		 // [Lc*8]  fine row of child (or -1)
		std::vector<int32_t>			  cidx;			 // [L]     8*parent_row + subidx (decoder replay)
		int								  Lc = 0;
		std::vector<std::vector<int32_t>> ncoarse;
		if( has_down ) {
			std::unordered_map<uint64_t, int32_t> fine_map;
			fine_map.reserve( ( size_t )L * 2 );
			for( int v = 0; v < L; ++v )
				fine_map[voxel_key( coords[( size_t )v * 3], coords[( size_t )v * 3 + 1], coords[( size_t )v * 3 + 2] )] = v;
			// unique parents
			std::unordered_map<uint64_t, int32_t> parent_seen;
			parent_seen.reserve( ( size_t )L );
			std::vector<std::array<int32_t, 3>> parents;
			for( int v = 0; v < L; ++v ) {
				const int32_t px = coords[( size_t )v * 3] >> 1, // // 2 (coords >= 0)
					py = coords[( size_t )v * 3 + 1] >> 1, pz = coords[( size_t )v * 3 + 2] >> 1;
				const uint64_t pk = voxel_key( px, py, pz );
				if( parent_seen.emplace( pk, 0 ).second )
					parents.push_back( { px, py, pz } );
			}
			std::sort( parents.begin(), parents.end() ); // lexicographic (x,y,z)
			Lc = ( int )parents.size();
			coarse_coords.resize( ( size_t )Lc * 3 );
			std::unordered_map<uint64_t, int32_t> coarse_map;
			coarse_map.reserve( ( size_t )Lc * 2 );
			for( int p = 0; p < Lc; ++p ) {
				coarse_coords[( size_t )p * 3]										 = parents[p][0];
				coarse_coords[( size_t )p * 3 + 1]									 = parents[p][1];
				coarse_coords[( size_t )p * 3 + 2]									 = parents[p][2];
				coarse_map[voxel_key( parents[p][0], parents[p][1], parents[p][2] )] = p;
			}
			// childidx[pr*8+o] = fine row of child (or -1 sentinel for missing)
			childidx.assign( ( size_t )Lc * 8, -1 );
			for( int p = 0; p < Lc; ++p ) {
				for( int o = 0; o < 8; ++o ) {
					const int32_t cx = 2 * parents[p][0] + ( o & 1 ), cy = 2 * parents[p][1] + ( ( o >> 1 ) & 1 ), cz = 2 * parents[p][2] + ( ( o >> 2 ) & 1 );
					auto		  it = fine_map.find( voxel_key( cx, cy, cz ) );
					if( it != fine_map.end() )
						childidx[( size_t )p * 8 + o] = it->second;
				}
			}
			// decoder replay: for each fine voxel, its coarse parent row + subidx
			cidx.resize( ( size_t )L );
			for( int v = 0; v < L; ++v ) {
				const int32_t px = coords[( size_t )v * 3] >> 1, py = coords[( size_t )v * 3 + 1] >> 1, pz = coords[( size_t )v * 3 + 2] >> 1;
				const int	  o	  = ( coords[( size_t )v * 3] & 1 ) | ( ( coords[( size_t )v * 3 + 1] & 1 ) << 1 ) | ( ( coords[( size_t )v * 3 + 2] & 1 ) << 2 );
				cidx[( size_t )v] = 8 * coarse_map[voxel_key( px, py, pz )] + o;
			}
			build_neighbor_indices( coarse_coords, Lc, ncoarse );
		}

		// ── voxel-count diagnostics ─────────────────────────────────────────
		// The shape decoder guards its levels against the backend's mul_mat
		// column ceiling (mul_mat_max_rows) and splits the finest one
		// (shape_dec_final_level_chunked). The encoder does neither: every
		// level is one graph over all L voxels, and its levels chain several
		// ConvNeXt blocks, so the decoder's split — which relies on the finest
		// level being a single conv — does not transfer.
		//
		// That makes a dense mesh's texture stage the place where a run dies,
		// which is what happens on ROCm today (see the progress note). Report the
		// counts BEFORE the graph runs, so the last line before an abort names
		// the level and its voxel count instead of leaving only a kernel error.
		{
			// The binding limit here is not the mul_mat column cap (measured for
			// a [64,64] weight; the encoder's convs are [64,128] and two runs
			// past 2^21 came out correct). It is that a GPU dispatch is sized in
			// WORKITEMS, in 32 bits, and this level launches several kernels
			// whose grid scales with the voxel count:
			//
			//   the S2C gather, get_rows over Lc*8 indices    -> Lc*8 * ~256
			//   the skip slices, one copy per channel group   -> C_out*Lc * ...
			//
			// The skip was the first to break and is fixed (strided slices in
			// place of ggml_mean); the gather is the next one. Fixing them one
			// at a time is a losing game — the level needs evaluating in
			// coarse-voxel blocks, the way shape_dec_level_chunked does it, and
			// blocks=0 here means it qualifies. Until then this reports the
			// counts rather than claiming to know which kernel goes first.
			// See docs/progress/1536-cascade_backend-limits.md.
			// Must mirror the condition below exactly. It has drifted twice
			// already, and a warning that describes a case which is in fact
			// handled sends the next investigation somewhere pointless.
			const bool	  will_chunk = has_down && !taps && ( int64_t )L > chunk_rows_limit();
			// This level's own mul_mat column ceiling. mul_mat_max_rows was
			// measured for a [64, 64] weight and the bug note says so; the
			// encoder's convs are wider, so treat it as the best estimate
			// available rather than a fact, and report where each level sits.
			const std::string cw	 = "blocks." + std::to_string( lvl ) + "." + std::to_string( hp.num_blocks[lvl] ) + ".conv1.weight";
			auto			  cwit	 = m->tensors.find( cw );
			const int64_t	  cap	 = cwit == m->tensors.end() ? mul_mat_max_rows( GGML_TYPE_F32 ) : mul_mat_max_rows( cwit->second->type );
			if( t2_timing )
				std::fprintf( stderr,
					"[shape_enc] level %d/%d: L=%d Lc=%d C_in=%d C_out=%d blocks=%d, S2C gather over %lld rows, mul_mat cap %lld%s%s\n",
					lvl,
					n_levels,
					L,
					Lc,
					C_in,
					C_out,
					hp.num_blocks[lvl],
					( long long )Lc * 8,
					( long long )cap,
					( int64_t )L > cap ? "  <-- PAST THE CAP" : "",
					will_chunk ? "  (chunked)" : "" );
			// Past the cap AND not split: the columns beyond it are never
			// written, so this level's latents are partly undefined. It does not
			// abort, which is the dangerous part — the corruption reaches the
			// texture flow, whose attention then spreads it across every token.
			// ...and only on a GPU. mul_mat_max_rows describes a ROCm/HIP kernel
			// ceiling; on the CPU backend there is none, and warning there sent
			// one investigation down the wrong path already.
			const bool on_cpu = m->backend_name.find( "CPU" ) != std::string::npos || m->backend_name.find( "cpu" ) != std::string::npos;
			if( !will_chunk && !on_cpu && ( int64_t )L > cap )
				std::fprintf( stderr,
					"[shape_enc] WARNING: level %d has %d voxels against a mul_mat column cap of %lld,"
					" and this level is not split (it has %d ConvNeXt blocks). Expect silently wrong"
					" latents past column %lld. See docs/progress/1536-cascade_backend-limits.md.\n",
					lvl,
					L,
					( long long )cap,
					hp.num_blocks[lvl],
					( long long )cap );
			// Only warn about what is actually still exposed. A warning that
			// keeps naming a condition already handled is worse than none — it
			// stops the next search at the wrong place, which this one did once.
			if( has_down && !will_chunk && ( int64_t )Lc > ( ( int64_t )1 << 21 ) )
				std::fprintf( stderr,
					"[shape_enc] WARNING: level %d has %d parent voxels, so its S2C gather spans %lld rows."
					" Past ~2^21 parents the per-row grids stop fitting a 32-bit dispatch and the launch is"
					" rejected. This level is not eligible for the block-wise path (level > 0, ConvNeXt"
					" blocks, or taps requested); expect an abort.\n",
					lvl,
					Lc,
					( long long )Lc * 8 );
		}

		// ── level 0 past the block size: evaluate it in voxel blocks ────────
		// Only level 0 ever gets large (Lc falls ~4x per level) and only level
		// 0 qualifies without a per-stage ping-pong, because it carries no
		// ConvNeXt blocks. Taps are excluded: they want the full-length
		// intermediates this path never forms.
		if( has_down && !taps && ( int64_t )L > chunk_rows_limit() ) {
			std::vector<float> lvl_out;
			if( !shape_enc_level_chunked( m, hp, lvl, C_in, C_out, L, Lc, lvl == 0 ? in_shift : feats, nfine, ncoarse, childidx, chunk_rows_limit(), lvl_out, error ) )
				return false;
			feats = std::move( lvl_out );
			// Same bookkeeping the single-graph path does at the end of a level
			// with a down-block: the decoder replays this in reverse order.
			trellis2_subdiv_level& sl = out_subs[( size_t )( n_levels - 2 - lvl )];
			sl.fine_coords			  = coords;
			sl.cidx					  = std::move( cidx );
			coords					  = std::move( coarse_coords );
			L						  = Lc;
			continue;
		}

		// ── graph ───────────────────────────────────────────────────────────
		const size_t	 gsize = 65536;
		const size_t	 mem   = ggml_tensor_overhead() * gsize + ggml_graph_overhead_custom( gsize, false );
		ggml_init_params ip { mem, nullptr, true };
		ggml_context*	 ctx = ggml_init( ip );
		ggml_cgraph*	 gf	 = ggml_new_graph_custom( ctx, gsize, false );

		auto			 W = [&]( const std::string& n ) -> ggml_tensor* {
			auto it = m->tensors.find( n );
			if( it == m->tensors.end() ) {
				if( missing.empty() )
					missing = n;
				return nullptr;
			}
			return it->second;
		};
		auto lin = [&]( ggml_tensor* in, const std::string& pfx ) -> ggml_tensor* {
			ggml_tensor* y = ggml_mul_mat( ctx, W( pfx + ".weight" ), in );
			ggml_tensor* b = W( pfx + ".bias" );
			if( b )
				y = ggml_add( ctx, y, b );
			return y;
		};
		auto ln_affine = [&]( ggml_tensor* h, const std::string& pfx ) -> ggml_tensor* {
			ggml_tensor* y = ggml_norm( ctx, h, eps );
			y			   = ggml_mul( ctx, y, W( pfx + ".weight" ) );
			y			   = ggml_add( ctx, y, W( pfx + ".bias" ) );
			return y;
		};

		// 27 fine + 27 coarse neighbor leaves
		std::vector<ggml_tensor*> idx_f( 27 ), mask_f( 27 ), idx_c( 27 ), mask_c( 27 );
		for( int k = 0; k < 27; ++k ) {
			idx_f[k]  = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, L );
			mask_f[k] = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, L );
			ggml_set_input( idx_f[k] );
			ggml_set_input( mask_f[k] );
		}
		if( has_down ) {
			for( int k = 0; k < 27; ++k ) {
				idx_c[k]  = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, Lc );
				mask_c[k] = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, Lc );
				ggml_set_input( idx_c[k] );
				ggml_set_input( mask_c[k] );
			}
		}
		auto conv = [&]( ggml_tensor* x, const std::string& pfx, std::vector<ggml_tensor*>& idxs, std::vector<ggml_tensor*>& masks ) -> ggml_tensor* {
			ggml_tensor* w = W( pfx + ".weight" ); // ne [Ci, 27, Co]
			ggml_tensor* b = W( pfx + ".bias" );
			if( !w || !b )
				return x;
			const int64_t Ci = w->ne[0], Co = w->ne[2];
			ggml_tensor*  acc = nullptr;
			for( int k = 0; k < 27; ++k ) {
				ggml_tensor* wk = ggml_cont( ctx, ggml_view_3d( ctx, w, Ci, 1, Co, w->nb[1], w->nb[2], ( size_t )k * w->nb[1] ) );
				wk				= ggml_reshape_2d( ctx, wk, Ci, Co );
				ggml_tensor* g	= ggml_get_rows( ctx, x, idxs[k] );
				g				= ggml_mul( ctx, g, masks[k] );
				ggml_tensor* y	= ggml_mul_mat( ctx, wk, g );
				acc				= acc ? ggml_add( ctx, acc, y ) : y;
			}
			return ggml_add( ctx, acc, b );
		};
		auto convnext = [&]( ggml_tensor* x, const std::string& pfx ) -> ggml_tensor* {
			ggml_tensor* h = conv( x, pfx + ".conv", idx_f, mask_f );
			h			   = ln_affine( h, pfx + ".norm" );
			h			   = lin( h, pfx + ".mlp.0" );
			h			   = ggml_silu( ctx, h );
			h			   = lin( h, pfx + ".mlp.2" );
			return ggml_add( ctx, h, x );
		};

		// input for this level: 6-channel dual grid at lvl 0 (through input_layer),
		// else the previous level's [C_in, L] features.
		const int	 in_dim = ( lvl == 0 ) ? hp.in_channels : C_in;
		ggml_tensor* in_a	= ggml_new_tensor_2d( ctx, GGML_TYPE_F32, in_dim, L );
		ggml_set_input( in_a );
		ggml_tensor* h = ( lvl == 0 ) ? lin( in_a, "input_layer" ) : in_a;

		for( int b = 0; b < hp.num_blocks[lvl]; ++b )
			h = convnext( h, "blocks." + std::to_string( lvl ) + "." + std::to_string( b ) );

		// S2C child-gather leaves (shared by the h1 and skip-x gathers)
		ggml_tensor *cidx_t = nullptr, *cmask_t = nullptr;
		ggml_tensor* out_h = nullptr;
		if( has_down ) {
			const std::string down = "blocks." + std::to_string( lvl ) + "." + std::to_string( hp.num_blocks[lvl] );
			cidx_t				   = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, ( int64_t )Lc * 8 );
			cmask_t				   = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, ( int64_t )Lc * 8 );
			ggml_set_input( cidx_t );
			ggml_set_input( cmask_t );

			ggml_tensor* hn1 = ggml_silu( ctx, ln_affine( h, down + ".norm1" ) );
			ggml_tensor* h1	 = conv( hn1, down + ".conv1", idx_f, mask_f ); // [C_out/8, L]
			// S2C gather: [ch, L] -> [ch, Lc*8] (masked) -> reshape [ch*8, Lc]
			auto		 s2c = [&]( ggml_tensor* t, int ch ) -> ggml_tensor* {
				ggml_tensor* g = ggml_get_rows( ctx, t, cidx_t ); // [ch, Lc*8]
				g			   = ggml_mul( ctx, g, cmask_t );
				return ggml_reshape_2d( ctx, ggml_cont( ctx, g ), ( int64_t )ch * 8, Lc );
			};
			ggml_tensor* h1c = s2c( h1, C_out / 8 );						 // [C_out, Lc]
			ggml_tensor* xc	 = s2c( h, C_in );								 // [C_in*8, Lc]
			ggml_tensor* hn2 = ggml_silu( ctx, ggml_norm( ctx, h1c, eps ) ); // norm2 affine-free
			ggml_tensor* h2	 = conv( hn2, down + ".conv2", idx_c, mask_c );	 // [C_out, Lc]
			// skip: mean over the (C_in*8 / C_out) group of xc, summed as gsz
			// strided slices rather than with ggml_mean.
			//
			// Same value; the reason is the launch. ggml_mean puts one
			// workgroup on each of its C_out*Lc rows with 32 threads apiece,
			// and a GPU dispatch is sized in workitems in 32 bits — so past
			// C_out*Lc*32 = 2^32 it is rejected outright and takes the process
			// down (measured: abort at Lc = 1,245,145, clean run at
			// Lc ~ 969,000; see the progress note). The slices below are plain copies
			// and adds over [C_out, Lc], which are flat-indexed and carry no
			// such ceiling, so the encoder keeps running on the GPU on meshes
			// where the mean cannot be launched at all.
			const int	 gsz  = ( C_in * 8 ) / C_out;
			ggml_tensor* g3	  = ggml_reshape_3d( ctx, xc, gsz, C_out, Lc );
			ggml_tensor* skip = nullptr;
			for( int j = 0; j < gsz; ++j ) {
				ggml_tensor* s = ggml_cont( ctx, ggml_view_3d( ctx, g3, 1, C_out, Lc, g3->nb[1], g3->nb[2], ( size_t )j * g3->nb[0] ) );
				skip		   = skip ? ggml_add( ctx, skip, s ) : s;
			}
			skip = ggml_scale( ctx, ggml_reshape_2d( ctx, skip, C_out, Lc ), 1.0f / ( float )gsz );
			out_h			  = ggml_add( ctx, h2, skip ); // [C_out, Lc]
		} else {
			ggml_tensor* hn = ggml_norm( ctx, h, 1e-5f ); // F.layer_norm affine-free
			ggml_tensor* z	= lin( hn, "to_latent" );	  // [2*latent, L]
			// take the mean half (posterior mean): channels [0, latent)
			out_h = ggml_cont( ctx, ggml_view_2d( ctx, z, hp.latent_channels, L, z->nb[1], 0 ) );
		}
		out_h = ggml_cont( ctx, out_h );
		ggml_set_output( out_h );
		ggml_build_forward_expand( gf, out_h );

		if( !missing.empty() ) {
			set_error( error, "missing tensor: " + missing + " (level " + std::to_string( lvl ) + ")" );
			ggml_free( ctx );
			return false;
		}

		ggml_gallocr_t alloc = ggml_gallocr_new( ggml_backend_get_default_buffer_type( m->backend ) );
		if( !ggml_gallocr_alloc_graph( alloc, gf ) ) {
			set_error( error, "ggml_gallocr_alloc_graph failed (enc level " + std::to_string( lvl ) + ")" );
			ggml_gallocr_free( alloc );
			ggml_free( ctx );
			return false;
		}

		// upload neighbor leaves (clamp missing to row 0, mask 0)
		auto upload_nbr = [&]( std::vector<std::vector<int32_t>>& nb, int Ln, std::vector<ggml_tensor*>& idxs, std::vector<ggml_tensor*>& masks ) {
			std::vector<int32_t> cl( ( size_t )Ln );
			std::vector<float>	 mk( ( size_t )Ln );
			for( int k = 0; k < 27; ++k ) {
				for( int v = 0; v < Ln; ++v ) {
					const bool miss = nb[k][( size_t )v] >= Ln;
					cl[( size_t )v] = miss ? 0 : nb[k][( size_t )v];
					mk[( size_t )v] = miss ? 0.0f : 1.0f;
				}
				ggml_backend_tensor_set( idxs[k], cl.data(), 0, ( size_t )Ln * sizeof( int32_t ) );
				ggml_backend_tensor_set( masks[k], mk.data(), 0, ( size_t )Ln * sizeof( float ) );
			}
		};
		upload_nbr( nfine, L, idx_f, mask_f );
		if( lvl == 0 )
			ggml_backend_tensor_set( in_a, in_shift.data(), 0, in_shift.size() * es );
		else
			ggml_backend_tensor_set( in_a, feats.data(), 0, feats.size() * es );
		if( has_down ) {
			upload_nbr( ncoarse, Lc, idx_c, mask_c );
			std::vector<int32_t> cl( ( size_t )Lc * 8 );
			std::vector<float>	 mk( ( size_t )Lc * 8 );
			for( size_t i = 0; i < cl.size(); ++i ) {
				const bool miss = childidx[i] < 0;
				cl[i]			= miss ? 0 : childidx[i];
				mk[i]			= miss ? 0.0f : 1.0f;
			}
			ggml_backend_tensor_set( cidx_t, cl.data(), 0, cl.size() * sizeof( int32_t ) );
			ggml_backend_tensor_set( cmask_t, mk.data(), 0, mk.size() * sizeof( float ) );
		}

		if( ggml_backend_graph_compute( m->backend, gf ) != GGML_STATUS_SUCCESS ) {
			set_error( error, "enc graph compute failed (level " + std::to_string( lvl ) + ")" );
			ggml_gallocr_free( alloc );
			ggml_free( ctx );
			return false;
		}

		const int C_next = has_down ? C_out : hp.latent_channels;
		const int L_next = has_down ? Lc : L;
		feats.resize( ( size_t )C_next * L_next );
		ggml_backend_tensor_get( out_h, feats.data(), 0, feats.size() * es );

		if( taps )
			cap( "enc_lvl" + std::to_string( lvl ), feats.data(), feats.size() );

		if( has_down ) {
			// record subdivision for the decoder (decoder order = reverse of encode)
			trellis2_subdiv_level& sl = out_subs[( size_t )( n_levels - 2 - lvl )];
			sl.fine_coords			  = coords; // this level's (fine) coords, in fine order
			sl.cidx					  = std::move( cidx );
			coords					  = std::move( coarse_coords );
			L						  = Lc;
		} else {
			out_slat   = feats;
			out_coords = coords;
		}
		ggml_gallocr_free( alloc );
		ggml_free( ctx );
	}
	return true;
}
