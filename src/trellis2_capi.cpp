// Flat C ABI wrapper around the trellis2 C++ API (see trellis2_capi.h).
//
// Owns the image-bytes decode (stb_image) so hosts can hand us untrusted
// uploads directly; everything past decode reuses the validated C++ pipeline:
//   decode -> preprocess -> DINOv3 -> SS-flow sample -> SS-dec -> marching cubes

#include "trellis2_capi.h"
#include "trellis2.h"
#include "cascade_tokens.h"
#include "mesh_export.h"
#include "print_remesh.h"
#include "pbr_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "marching_cubes.h"
#include "flexible_dual_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{

void copy_err( char* err, int err_len, const std::string& msg )
{
	if( !err || err_len <= 0 )
		return;
	std::snprintf( err, ( size_t )err_len, "%s", msg.c_str() );
}

// Rough peak VRAM the shape decode's transient buffers need at each tier
// (measured on the reference image: ~2.4 GB at 512³, ~6.75 GB for the 1024³
// level-3 conv output; rounded up for headroom). Mesh-dependent, so treated as
// a threshold, not an exact reservation — the free-flows fallback then gives a
// several-GB cushion if the estimate is low.
//
// The 1536 entry is NOT measured: it is surface-area-scaled from the 1024 tier
// (~9M voxels, ~15 GB) per docs/plan/1536-cascade.md D4, and Phase 3 of that
// plan replaces it with a real number. It is only ever compared against free
// VRAM to decide whether to free the finished flow DiTs first, so erring high
// costs a reload, not a failure.
size_t decode_vram_peak( int pipeline_type )
{
	const double GB = ( double )( 1ULL << 30 );
	double		 gb = 3.0; // 512³
	if( pipeline_type == T2_PIPE_1024 )
		gb = 7.5;
	else if( pipeline_type == T2_PIPE_1536 )
		gb = 15.0;
	return ( size_t )( gb * GB );
}

// Both cascade tiers run the same HR flow (the 1024 model) on the same
// conditioning and therefore share every model-selection decision; they differ
// only in the scaffold resolution the token budget lands on.
bool is_cascade_pipeline( int pipeline_type )
{
	return pipeline_type == T2_PIPE_1024 || pipeline_type == T2_PIPE_1536;
}

// The resolution a cascade tier asks the token budget for.
int requested_cascade_resolution( int pipeline_type )
{
	return pipeline_type == T2_PIPE_1536 ? 1536 : 1024;
}

// TRELLIS.2's HR token budget. T2_MAX_NUM_TOKENS overrides it, because the
// default was chosen for cards much smaller than the one the 1536 tier targets
// and docs/plan/1536-cascade.md Phase 3 has to measure before that default can
// responsibly move. Unset, this reproduces upstream exactly.
int cascade_max_tokens()
{
	if( const char* e = std::getenv( "T2_MAX_NUM_TOKENS" ) ) {
		const int v = std::atoi( e );
		if( v > 0 )
			return v;
	}
	return t2cascade::default_max_num_tokens;
}

} // namespace

struct t2_pipeline {
	trellis2_dino_model*	  dino	   = nullptr;
	trellis2_ss_flow_model*	  flow	   = nullptr;
	trellis2_ss_dec_model*	  dec	   = nullptr;
	trellis2_slat_flow_model* slat	   = nullptr; // 512 model (fine path)
	trellis2_slat_flow_model* slat_hr  = nullptr; // 1024 model (cascade)
	trellis2_shape_dec_model* shapedec = nullptr; // shared by 512 + cascade
	std::string				  backend;
	bool					  fine		   = false; // 512 dual-grid available
	bool					  cascade	   = false; // 1024 cascade available
	bool					  texture	   = false; // PBR texturing available (shape_enc + tex_dec + tex_flow)
	int						  flags		   = 0;
	bool					  shapedec_gpu = false; // shape decoder placed on the GPU (VRAM permitting)
	// gguf paths, so the flow DiTs can be freed to make VRAM room for a GPU
	// decode and lazily reloaded on the next generate (see ensure_decode_vram).
	std::string				  ss_flow_path, slat_path, slat_hr_path;
	// Texture-stage gguf paths. The tex models (~4 GB) are loaded lazily inside
	// the texture stage — after the geometry flow DiTs are freed — and freed
	// again, so they never coexist in VRAM with the geometry flows.
	std::string				  shapeenc_path, texdec_path, texflow_path, texflow_hr_path;
};

// A generated mesh: verts (3/vertex), normals (3/vertex), tris (3/tri), and
// optional per-vertex PBR (6/vertex: base_color rgb, metallic, roughness, alpha).
struct t2_mesh_result {
	std::vector<float>	 verts;
	std::vector<float>	 normals;
	std::vector<int>	 tris;
	std::vector<float>	 pbr; // empty when untextured
	// Only filled by t2_prepare_quad_mesh; read back with t2_quad_mesh_stats.
	// The struct is opaque to hosts, so growing it costs them nothing.
	t2glb::QuadMeshStats quad_stats;
	bool				 has_quad_stats = false;
	// Grid the geometry was extracted at (t2_mesh_grid_resolution). Set only by
	// t2_generate; 0 means "not from a generation".
	int					 grid_res		= 0;
};

namespace
{

// Reload any flow DiT freed by a previous GPU decode (see ensure_decode_vram).
// No-op the common case where nothing was freed (pointers still set).
bool reload_flows( t2_pipeline* p, std::string& e )
{
	if( !p->flow && !p->ss_flow_path.empty() ) {
		p->flow = trellis2_ss_flow_load( p->ss_flow_path.c_str(), true, &e );
		if( !p->flow )
			return false;
	}
	if( !p->slat && !p->slat_path.empty() ) {
		p->slat = trellis2_slat_flow_load( p->slat_path.c_str(), true, &e );
		if( !p->slat )
			return false;
	}
	if( !p->slat_hr && !p->slat_hr_path.empty() ) {
		p->slat_hr = trellis2_slat_flow_load( p->slat_hr_path.c_str(), true, &e );
		if( !p->slat_hr )
			return false;
	}
	return true;
}

// Before a GPU shape decode, make room: if the decode's transient buffers would
// not fit in current free VRAM, free the flow DiTs (all finished by decode time)
// to reclaim their ~5-7 GB. reload_flows() brings them back on the next
// generate. No-op for a CPU decoder or when the decode already fits.
void ensure_decode_vram( t2_pipeline* p, int pipeline_type )
{
	if( !p->shapedec_gpu )
		return;
	if( trellis2_gpu_free_vram() >= decode_vram_peak( pipeline_type ) )
		return;
	trellis2_ss_flow_free( p->flow );
	p->flow = nullptr;
	trellis2_slat_flow_free( p->slat );
	p->slat = nullptr;
	trellis2_slat_flow_free( p->slat_hr );
	p->slat_hr = nullptr;
}

// ── live intermediate 3D previews (voxel sets) ───────────────────────────────
//
// A preview is a self-describing "T2VOX01" blob the host streams to its viewer:
//   char magic[8] = "T2VOX01\0"; u32 res; u32 nvox; u16 coords[3*nvox] in [0,res)
// Voxels bound the payload (occupancy is sparse) and read as clearly "coarse"
// against the final smooth mesh — the stages stay visually distinct.

// Collect the occupied cells (logit > 0) of a dense [res^3] occupancy grid.
void collect_occupied( const float* occ, int res, std::vector<int32_t>& cells )
{
	cells.clear();
	for( int x = 0; x < res; ++x )
		for( int y = 0; y < res; ++y )
			for( int z = 0; z < res; ++z ) {
				if( occ[( ( size_t )x * res + y ) * res + z] > 0.0f ) {
					cells.push_back( x );
					cells.push_back( y );
					cells.push_back( z );
				}
			}
}

// Pack a flat [x,y,z,...] cell list (each in [0,res)) into a T2VOX01 blob.
void pack_voxels( int res, const std::vector<int32_t>& cells, std::vector<uint8_t>& blob )
{
	const uint32_t nvox = ( uint32_t )( cells.size() / 3 );
	const uint32_t r	= ( uint32_t )res;
	blob.resize( 8 + 4 + 4 + ( size_t )nvox * 3 * 2 );
	std::memcpy( blob.data(), "T2VOX01", 8 ); // 7 chars + NUL
	std::memcpy( blob.data() + 8, &r, 4 );
	std::memcpy( blob.data() + 12, &nvox, 4 );
	uint8_t* dst = blob.data() + 16;
	for( uint32_t i = 0; i < nvox * 3; ++i ) {
		const uint16_t c = ( uint16_t )cells[i];
		std::memcpy( dst, &c, 2 );
		dst += 2;
	}
}

// Pack + hand a voxel preview to the host callback. Best-effort: never fatal.
void emit_voxels( t2_preview_fn fn, void* user, int stage, int step, int total, int res, const std::vector<int32_t>& cells )
{
	if( !fn )
		return;
	std::vector<uint8_t> blob;
	pack_voxels( res, cells, blob );
	fn( user, stage, step, total, blob.data(), ( int )blob.size() );
}

// ── intermediate shape-SLAT mesh keyframes (opt-in) ──────────────────────────
//
// A keyframe is a coarse marching-cubes mesh (T2MESH01 blob, same layout as the
// final /api/mesh) of an intermediate shape-flow x_0 estimate — so the surface
// can be watched forming during the otherwise-invisible SLAT stage. Decoding
// mid-flow would OOM the 16 GB budget (the flow DiT is resident and can't be
// freed while its sampler is running), so we CAPTURE the cheap latents during
// the flow and REPLAY them into meshes AFTER the final decode, when the flow
// DiTs are freed and the shape decoder owns VRAM.

// Pack an MC mesh into a T2MESH01 blob: magic[8] u32 nv u32 nt
//   f32[3nv] verts  f32[3nv] normals  i32[3nt] tris  (little-endian).
void pack_mesh( const mc::Mesh& m, std::vector<uint8_t>& blob )
{
	const uint32_t nv = ( uint32_t )( m.verts.size() / 3 );
	const uint32_t nt = ( uint32_t )( m.tris.size() / 3 );
	blob.resize( 16 + ( size_t )nv * 24 + ( size_t )nt * 12 );
	std::memcpy( blob.data(), "T2MESH01", 8 );
	std::memcpy( blob.data() + 8, &nv, 4 );
	std::memcpy( blob.data() + 12, &nt, 4 );
	size_t o = 16;
	std::memcpy( blob.data() + o, m.verts.data(), ( size_t )nv * 12 );
	o += ( size_t )nv * 12;
	std::memcpy( blob.data() + o, m.normals.data(), ( size_t )nv * 12 );
	o += ( size_t )nv * 12;
	std::memcpy( blob.data() + o, m.tris.data(), ( size_t )nt * 12 );
}

// One shape-flow stage's captured intermediate x_0 latents (denormalized) plus
// the scaffold they sit on, awaiting post-decode replay.
struct kf_capture {
	int								stage	  = 0; // T2_STAGE_SLAT_FLOW or T2_STAGE_SLAT_FLOW_HR
	int								res_in	  = 0; // scaffold resolution (32 LR / 64 HR)
	int								levels	  = 0; // upsample levels to the ~128^3 keyframe grid
	int								stride	  = 1; // capture every `stride` steps (plus the last)
	int								channels  = 0; // latent channels (32)
	const float*					norm_mean = nullptr;
	const float*					norm_std  = nullptr;
	std::vector<int32_t>			coords;		   // scaffold coords, copied once
	std::vector<int>				steps, totals; // per capture, for labelling
	std::vector<std::vector<float>> latents;	   // denormalized [L*channels] each
};

// Sampler preview trampoline: denormalize + stash the step's x_0 estimate.
void kf_capture_cb( void* u, int step, int total, const float* latent, int n )
{
	auto* c = ( kf_capture* )u;
	if( c->stride < 1 )
		return;
	if( step != total && ( step % c->stride ) != 0 )
		return; // stride, plus the last step
	const int		   C = c->channels;
	std::vector<float> den( ( size_t )n );
	for( int i = 0; i < n; ++i )
		den[i] = latent[i] * c->norm_std[i % C] + c->norm_mean[i % C];
	c->steps.push_back( step );
	c->totals.push_back( total );
	c->latents.push_back( std::move( den ) );
}

// Replay captured latents into coarse MC-mesh keyframes and stream each. Runs
// after the final decode (decoder owns VRAM). Best-effort: skips on any failure.
void emit_keyframes( trellis2_shape_dec_model* dec, kf_capture& kf, t2_preview_fn fn, void* user )
{
	if( !fn || kf.latents.empty() || !dec )
		return;
	const int L	  = ( int )( kf.coords.size() / 3 );
	const int tgt = kf.res_in << kf.levels; // keyframe grid (128^3)
	for( size_t k = 0; k < kf.latents.size(); ++k ) {
		std::vector<int32_t> up;
		std::string			 e;
		if( !trellis2_shape_dec_upsample( dec, kf.latents[k].data(), L, kf.coords.data(), kf.levels, up, &e ) )
			continue;
		std::vector<float> field( ( size_t )tgt * tgt * tgt, 0.0f );
		for( size_t i = 0; i + 2 < up.size(); i += 3 ) {
			const int x = up[i], y = up[i + 1], z = up[i + 2];
			if( x >= 0 && x < tgt && y >= 0 && y < tgt && z >= 0 && z < tgt )
				field[( ( size_t )x * tgt + y ) * tgt + z] = 1.0f;
		}
		mc::Mesh m = mc::extract( field.data(), tgt, tgt, tgt, 0.5f );
		if( m.verts.empty() )
			continue;
		const float inv = 1.0f / ( float )tgt;
		for( auto& v : m.verts )
			v = v * inv - 0.5f; // -> centered unit cube
		std::vector<uint8_t> blob;
		pack_mesh( m, blob );
		fn( user, kf.stage, kf.steps[k], kf.totals[k], blob.data(), ( int )blob.size() );
	}
}

// Per-step SS preview cadence: T2_PREVIEW_STRIDE if set, else ~4 across the run.
int preview_stride( int steps )
{
	if( const char* e = std::getenv( "T2_PREVIEW_STRIDE" ) ) {
		const int s = std::atoi( e );
		if( s > 0 )
			return s;
	}
	return std::max( 1, steps / 4 );
}

// Context for the (capture-less) SS-sampler preview trampoline: decode the
// handed-out x_0 estimate into a 64^3 occupancy and stream it as voxels.
struct ss_preview_ctx {
	t2_preview_fn		   fn	  = nullptr;
	void*				   user	  = nullptr;
	trellis2_ss_dec_model* dec	  = nullptr;
	int					   res	  = 0;
	int					   stride = 1;
	std::vector<float>*	   occ	  = nullptr; // scratch [res^3], caller-owned
};

// PBR-texture stage on the freshly decoded dual grid. Encoding the decoder
// output again is less direct than replaying the shape decoder's subdivisions,
// but it is the numerically validated path used by test_texture. In particular,
// it avoids the all-saturated material regression seen with the first integrated
// subdivision-guide implementation. The decoded PBR volume is still sampled at
// the actual mesh vertices rather than copied by voxel index.
bool run_texture_stage( t2_pipeline* p,
	const std::vector<float>&		 dec_feats,
	const std::vector<int32_t>&		 dec_coords,
	const std::vector<float>&		 mesh_verts,
	int								 grid,
	int								 pt,
	const trellis2_dino_cond&		 cond,
	uint64_t						 seed,
	int								 texture_steps,
	t2_progress_fn					 progress,
	void*							 user,
	std::vector<float>&				 pbr_out,
	std::string&					 e )
{
	const int nvox = ( int )( dec_coords.size() / 3 );
	if( nvox <= 0 || dec_feats.size() != ( size_t )nvox * 7 ) {
		e = "invalid decoded dual grid";
		return false;
	}

	// Shape-encoder input from the seven-channel dual grid: learned dual-vertex
	// offset plus the three intersection flags. This matches the standalone
	// texturing pipeline fixture and preserves its exact subdivision replay.
	const float		   mg = 0.5f;
	std::vector<float> in6( ( size_t )nvox * 6 );
	for( int v = 0; v < nvox; ++v ) {
		const float* f = dec_feats.data() + ( size_t )v * 7;
		for( int c = 0; c < 3; ++c ) {
			const float s				 = 1.0f / ( 1.0f + std::exp( -f[c] ) );
			in6[( size_t )v * 6 + c]	 = ( 1.0f + 2.0f * mg ) * s - mg;
			in6[( size_t )v * 6 + 3 + c] = f[3 + c] > 0.0f ? 1.0f : 0.0f;
		}
	}

	if( progress )
		progress( user, T2_STAGE_TEXTURE, 0, texture_steps > 0 ? texture_steps : 12 );

	// TRELLIS2_SHAPE_ENC_CPU forces the encoder onto the CPU. Not needed for the
	// dispatch-limit crash any more — that is fixed inside the encoder, where
	// the skip mean is now summed as strided slices instead of one workgroup
	// per row — but kept as an escape hatch, mirroring TRELLIS2_SHAPE_DEC_CPU.
	const char*				  enc_device = std::getenv( "TRELLIS2_SHAPE_ENC_CPU" ) ? "cpu" : nullptr;
	trellis2_shape_enc_model* enc		 = trellis2_shape_enc_load( p->shapeenc_path.c_str(), true, &e, enc_device );
	if( !enc ) {
		e = "shape_enc load: " + e;
		return false;
	}
	std::vector<float>				   shape_slat;
	std::vector<int32_t>			   lat_coords;
	std::vector<trellis2_subdiv_level> subs;
	bool							   ok = trellis2_shape_enc_encode( enc, in6.data(), nvox, dec_coords.data(), shape_slat, lat_coords, subs, nullptr, &e );
	trellis2_shape_enc_free( enc );
	if( !ok ) {
		e = "shape encode: " + e;
		return false;
	}

	const int Nl = ( int )( lat_coords.size() / 3 );
	if( Nl <= 0 || shape_slat.size() != ( size_t )Nl * 32 ) {
		e = "invalid generated shape SLAT";
		return false;
	}
	if( subs.empty() ) {
		e = "missing shape decoder subdivision guide";
		return false;
	}

	// A cascade shape SLAT (64^3 and up) must use the 1024 material model;
	// feeding it to the 32^3/512 model is not a valid fallback. Upstream picks
	// tex_slat_flow_model_1024 for the 1536 tier too — the texture stage is
	// resolution-agnostic in the same way the shape HR flow is.
	if( is_cascade_pipeline( pt ) && p->texflow_hr_path.empty() ) {
		e = "1024 texture flow model is not loaded";
		return false;
	}
	const std::string&		  fp   = is_cascade_pipeline( pt ) ? p->texflow_hr_path : p->texflow_path;
	trellis2_slat_flow_model* flow = trellis2_slat_flow_load( fp.c_str(), true, &e );
	if( !flow ) {
		e = "tex_flow load: " + e;
		return false;
	}
	std::vector<float>		   tex_slat( ( size_t )Nl * 32 );
	trellis2_ss_sampler_params tp; // texturing_pipeline.json tex_slat_sampler
	tp.steps				 = texture_steps > 0 ? texture_steps : 12;
	tp.guidance_strength	 = 1.0f;
	tp.guidance_rescale		 = 0.0f;
	tp.guidance_interval_min = 0.6f;
	tp.guidance_interval_max = 0.9f;
	tp.rescale_t			 = 3.0f;
	tp.seed					 = seed ^ 0x7e00ULL;
	tp.verbose				 = false;
	struct tex_progress_ctx {
		t2_progress_fn fn;
		void*		   user;
	} pc { progress, user };
	if( progress ) {
		progress( user, T2_STAGE_TEXTURE, 0, tp.steps );
		tp.progress = []( void* u, int step, int total ) {
			auto* c = ( tex_progress_ctx* )u;
			c->fn( c->user, T2_STAGE_TEXTURE, step, total );
		};
		tp.progress_user = &pc;
	}
	ok = trellis2_slat_flow_sample_tex(
		flow, Nl, lat_coords.data(), cond.data.data(), ( int )cond.tokens(), ( int )cond.channels(), shape_slat.data(), &tp, nullptr, /*denormalize*/ true, tex_slat.data(), &e );
	trellis2_slat_flow_free( flow );
	if( !ok ) {
		e = "tex sample: " + e;
		return false;
	}

	// Texture decoder -> sparse six-channel PBR, replaying the shape decoder's
	// subdivision hierarchy (upstream's guide_subs).
	trellis2_shape_dec_model* texdec = trellis2_tex_dec_load( p->texdec_path.c_str(), true, &e );
	if( !texdec ) {
		e = "tex_dec load: " + e;
		return false;
	}
	std::vector<float>	 pbr;
	std::vector<int32_t> pbr_coords;
	ok = trellis2_tex_dec_decode( texdec, tex_slat.data(), Nl, lat_coords.data(), subs, pbr, pbr_coords, &e );
	trellis2_shape_dec_free( texdec );
	if( !ok ) {
		e = "tex decode: " + e;
		return false;
	}

	// Sample at the actual dual-grid vertex positions, not their parent voxel
	// coordinates. q = (xyz + 0.5) * grid is upstream's MeshWithVoxel query.
	const int M = ( int )( pbr_coords.size() / 3 );
	if( M <= 0 || pbr.size() != ( size_t )M * 6 ) {
		e = "texture decoder returned an invalid PBR volume";
		return false;
	}
	const int		   nv = ( int )( mesh_verts.size() / 3 );
	std::vector<float> query( ( size_t )nv * 3 ), weights( ( size_t )nv );
	for( int v = 0; v < nv; ++v )
		for( int c = 0; c < 3; ++c )
			query[( size_t )v * 3 + c] = ( mesh_verts[( size_t )v * 3 + c] + 0.5f ) * grid;
	pbr_out.resize( ( size_t )nv * 6 );
	t2pbr::sample_sparse_trilinear( pbr.data(), M, 6, pbr_coords.data(), query.data(), nv, pbr_out.data(), weights.data() );
	for( int v = 0; v < nv; ++v ) {
		float* d = pbr_out.data() + ( size_t )v * 6;
		if( weights[( size_t )v] <= 1e-6f ) {
			d[0] = d[1] = d[2] = 0.5f;
			d[3]			   = 0.0f;
			d[4]			   = 0.5f;
			d[5]			   = 1.0f;
		} else {
			for( int c = 0; c < 6; ++c )
				d[c] = std::max( 0.0f, std::min( 1.0f, d[c] ) );
		}
	}

	// An opaque material can legitimately be white, metallic, or rough, but a
	// decoder result with essentially every one of the six channels pinned to
	// one is a collapsed latent, not a usable texture. Never silently persist
	// that as a successful textured generation again.
	if( t2pbr::is_collapsed_saturated( pbr_out.data(), nv ) ) {
		pbr_out.clear();
		e = "texture decoder produced a collapsed saturated material";
		return false;
	}
	return true;
}

} // namespace

extern "C" {

int t2_abi_version()
{
	return T2_CAPI_ABI_VERSION;
}

t2_pipeline* t2_pipeline_load( const char* dino_gguf,
	const char*							   ss_flow_gguf,
	const char*							   ss_dec_gguf,
	const char*							   slat_flow_gguf,
	const char*							   slat_hr_flow_gguf,
	const char*							   shape_dec_gguf,
	const char*							   shape_enc_gguf,
	const char*							   tex_dec_gguf,
	const char*							   tex_flow_gguf,
	const char*							   tex_flow_hr_gguf,
	int									   flags,
	char*								   err,
	int									   err_len )
{
	std::string e;
	auto*		p = new t2_pipeline();
	p->flags	  = flags;
	p->dino		  = trellis2_dino_load( dino_gguf, true, &e );
	if( !p->dino ) {
		copy_err( err, err_len, "dino: " + e );
		t2_pipeline_free( p );
		return nullptr;
	}
	// Free VRAM before the flow DiTs are loaded == the VRAM reclaimable by
	// freeing them again at decode time. Drives the shape-decoder placement.
	const size_t free_pre_flows = trellis2_gpu_free_vram();
	p->flow						= trellis2_ss_flow_load( ss_flow_gguf, true, &e );
	if( !p->flow ) {
		copy_err( err, err_len, "ss_flow: " + e );
		t2_pipeline_free( p );
		return nullptr;
	}
	// The SS occupancy decoder uses a genuine dense CONV_3D, for which ggml has
	// no CUDA kernel, so it stays on the CPU (it is only ~3 s / 4 % anyway).
	p->dec = trellis2_ss_dec_load( ss_dec_gguf, true, &e, "cpu" );
	if( !p->dec ) {
		copy_err( err, err_len, "ss_dec: " + e );
		t2_pipeline_free( p );
		return nullptr;
	}

	auto present = []( const char* s ) { return s && s[0]; };

	if( present( slat_flow_gguf ) && present( shape_dec_gguf ) ) {
		p->slat = trellis2_slat_flow_load( slat_flow_gguf, true, &e );
		if( !p->slat ) {
			copy_err( err, err_len, "slat_flow: " + e );
			t2_pipeline_free( p );
			return nullptr;
		}
		const bool will_cascade = present( slat_hr_flow_gguf );

		// Remember the flow-DiT gguf paths so a GPU decode can free them for VRAM
		// and reload them next generate (ensure_decode_vram / reload_flows).
		p->ss_flow_path = ss_flow_gguf ? ss_flow_gguf : "";
		p->slat_path	= slat_flow_gguf;
		p->slat_hr_path = will_cascade ? slat_hr_flow_gguf : "";

		// Auto-place the shape (FlexiDualGrid VAE) decoder — the biggest fine-path
		// stage (~44 s CPU / 59 %). Its mask-based submanifold conv runs ~20x
		// faster on the GPU (~2 s), but the decode's transient buffers need the
		// flow DiTs' VRAM freed first (done per-request in ensure_decode_vram).
		// Put it on the GPU when the card can hold that decode once the flows are
		// freed — i.e. free_pre_flows covers the tier's decode peak plus the
		// decoder weights (~1 GB) and a margin. TRELLIS2_SHAPE_DEC_{GPU,CPU}
		// force it. On the CPU (no GPU) or too small a card it stays on the CPU,
		// so there is never a mid-generation OOM.
		bool sd_gpu;
		if( std::getenv( "TRELLIS2_SHAPE_DEC_CPU" ) )
			sd_gpu = false;
		else if( std::getenv( "TRELLIS2_SHAPE_DEC_GPU" ) )
			sd_gpu = free_pre_flows > 0;
		else if( free_pre_flows == 0 )
			sd_gpu = false; // no GPU
		else {
			const size_t margin = ( size_t )3 << 29; // ~1.5 GB (weights + slack)
			sd_gpu				= free_pre_flows >= decode_vram_peak( will_cascade ? T2_PIPE_1024 : T2_PIPE_512 ) + margin;
		}
		p->shapedec = trellis2_shape_dec_load( shape_dec_gguf, true, &e, sd_gpu ? nullptr : "cpu" );
		if( !p->shapedec && sd_gpu ) { // unexpected GPU load OOM — fall back to CPU
			sd_gpu		= false;
			p->shapedec = trellis2_shape_dec_load( shape_dec_gguf, true, &e, "cpu" );
		}
		if( !p->shapedec ) {
			copy_err( err, err_len, "shape_dec: " + e );
			t2_pipeline_free( p );
			return nullptr;
		}
		p->shapedec_gpu = sd_gpu;
		p->fine			= true;

		// The 1024 model is optional; when present the cascade path is enabled
		// and reuses p->shapedec for both the upsample and the 1024^3 decode.
		if( will_cascade ) {
			p->slat_hr = trellis2_slat_flow_load( slat_hr_flow_gguf, true, &e );
			if( !p->slat_hr ) {
				copy_err( err, err_len, "slat_hr_flow: " + e );
				t2_pipeline_free( p );
				return nullptr;
			}
			p->cascade = true;
		}

		// PBR texturing: enabled when the shape encoder, texture decoder, and
		// (at least the 512) texture flow are present. The tex models are loaded
		// lazily per-generate (run_texture_stage), so only their paths are kept.
		if( present( shape_enc_gguf ) && present( tex_dec_gguf ) && present( tex_flow_gguf ) ) {
			std::string				  ve;
			trellis2_shape_enc_model* te = trellis2_shape_enc_load( shape_enc_gguf, false, &ve );
			if( !te ) {
				copy_err( err, err_len, "shape_enc: " + ve );
				t2_pipeline_free( p );
				return nullptr;
			}
			trellis2_shape_enc_free( te );
			trellis2_shape_dec_model* td = trellis2_tex_dec_load( tex_dec_gguf, false, &ve );
			if( !td ) {
				copy_err( err, err_len, "tex_dec: " + ve );
				t2_pipeline_free( p );
				return nullptr;
			}
			if( trellis2_shape_dec_hparams_of( td ).out_channels != 6 ) {
				trellis2_shape_dec_free( td );
				copy_err( err, err_len, "tex_dec: expected 6 output channels" );
				t2_pipeline_free( p );
				return nullptr;
			}
			trellis2_shape_dec_free( td );
			trellis2_slat_flow_model* tf = trellis2_slat_flow_load( tex_flow_gguf, false, &ve );
			if( !tf ) {
				copy_err( err, err_len, "tex_flow: " + ve );
				t2_pipeline_free( p );
				return nullptr;
			}
			if( trellis2_slat_flow_hparams_of( tf ).concat_cond_channels != 32 ) {
				trellis2_slat_flow_free( tf );
				copy_err( err, err_len, "tex_flow: expected 32 concat-conditioning channels" );
				t2_pipeline_free( p );
				return nullptr;
			}
			trellis2_slat_flow_free( tf );
			if( present( tex_flow_hr_gguf ) ) {
				tf = trellis2_slat_flow_load( tex_flow_hr_gguf, false, &ve );
				if( !tf ) {
					copy_err( err, err_len, "tex_flow_hr: " + ve );
					t2_pipeline_free( p );
					return nullptr;
				}
				if( trellis2_slat_flow_hparams_of( tf ).concat_cond_channels != 32 ) {
					trellis2_slat_flow_free( tf );
					copy_err( err, err_len, "tex_flow_hr: expected 32 concat-conditioning channels" );
					t2_pipeline_free( p );
					return nullptr;
				}
				trellis2_slat_flow_free( tf );
			}
			p->shapeenc_path   = shape_enc_gguf;
			p->texdec_path	   = tex_dec_gguf;
			p->texflow_path	   = tex_flow_gguf;
			p->texflow_hr_path = present( tex_flow_hr_gguf ) ? tex_flow_hr_gguf : "";
			p->texture		   = true;
		}
	}

	p->backend = trellis2_ss_flow_backend_name( p->flow );
	return p;
}

int t2_pipeline_caps( t2_pipeline* p )
{
	if( !p )
		return 0;
	int c = T2_CAP_COARSE;
	if( p->fine )
		c |= T2_CAP_512;
	// One flag, two tiers: 1536 needs no model 1024 does not already need.
	if( p->cascade )
		c |= T2_CAP_1024 | T2_CAP_1536;
	if( p->texture )
		c |= T2_CAP_TEXTURE;
	return c;
}

int t2_pipeline_is_fine( t2_pipeline* p )
{
	return p && ( p->fine || p->cascade ) ? 1 : 0;
}

void t2_pipeline_free( t2_pipeline* p )
{
	if( !p )
		return;
	trellis2_dino_free( p->dino );
	trellis2_ss_flow_free( p->flow );
	trellis2_ss_dec_free( p->dec );
	trellis2_slat_flow_free( p->slat );
	trellis2_slat_flow_free( p->slat_hr );
	trellis2_shape_dec_free( p->shapedec );
	delete p;
}

const char* t2_pipeline_backend( t2_pipeline* p )
{
	return p ? p->backend.c_str() : "none";
}

static int preprocess_image_bytes_mode( const void* image_bytes, int image_len, int out_size, unsigned char* out_rgb, int background_mode, char* err, int err_len )
{
	if( !image_bytes || image_len <= 0 || out_size <= 0 || !out_rgb ) {
		copy_err( err, err_len, "invalid arguments" );
		return 1;
	}
	// Reject absurd dimensions before decoding (upload DoS / decompression
	// bombs): 16 MPixel is far beyond anything useful at a 512^2 target.
	int w = 0, h = 0, comp = 0;
	if( !stbi_info_from_memory( ( const stbi_uc* )image_bytes, image_len, &w, &h, &comp ) ) {
		copy_err( err, err_len, std::string( "image probe failed: " ) + stbi_failure_reason() );
		return 1;
	}
	if( w <= 0 || h <= 0 || ( int64_t )w * h > ( int64_t )16 * 1024 * 1024 ) {
		copy_err( err, err_len, "image dimensions out of range" );
		return 1;
	}
	unsigned char* px = stbi_load_from_memory( ( const stbi_uc* )image_bytes, image_len, &w, &h, &comp, 4 );
	if( !px ) {
		copy_err( err, err_len, std::string( "image decode failed: " ) + stbi_failure_reason() );
		return 1;
	}

	// stb supplies opaque alpha for inputs without it and preserves alpha for
	// grayscale+alpha, RGBA, and transparent palette images. Turn an otherwise
	// opaque border-connected near-black/near-white background into alpha before
	// the reference alpha-bbox crop; meaningful existing masks are trusted.
	if( background_mode < T2_BACKGROUND_AUTO || background_mode > T2_BACKGROUND_WHITE ) {
		stbi_image_free( px );
		copy_err( err, err_len, "invalid background mode" );
		return 1;
	}
	trellis2_remove_solid_background_rgba( px, w, h, background_mode );

	std::string			 e;
	std::vector<uint8_t> rgb;
	const bool			 ok = trellis2_preprocess_rgba( px, w, h, out_size, rgb, &e );
	stbi_image_free( px );
	if( !ok ) {
		copy_err( err, err_len, "preprocess failed: " + e );
		return 1;
	}
	std::memcpy( out_rgb, rgb.data(), rgb.size() );
	return 0;
}

int t2_preprocess_image_bytes( const void* image_bytes, int image_len, int out_size, unsigned char* out_rgb, char* err, int err_len )
{
	return preprocess_image_bytes_mode( image_bytes, image_len, out_size, out_rgb, T2_BACKGROUND_AUTO, err, err_len );
}

t2_mesh_result* t2_generate( t2_pipeline* p,
	const void*							  image_bytes,
	int									  image_len,
	int									  pipeline_type,
	int									  background_mode,
	uint64_t							  seed,
	int									  steps,
	float								  guidance,
	int									  texture_steps,
	t2_progress_fn						  progress,
	void*								  user,
	t2_preview_fn						  preview,
	void*								  preview_user,
	char*								  err,
	int									  err_len )
{
	if( !p ) {
		copy_err( err, err_len, "null pipeline" );
		return nullptr;
	}
	std::string e;

	// Reload any flow DiT a previous GPU decode freed for VRAM (usually a no-op).
	if( !reload_flows( p, e ) ) {
		copy_err( err, err_len, "reload flow models: " + e );
		return nullptr;
	}

	// Resolve the requested path to what is actually loaded. AUTO stops at 1024
	// on purpose: 1536 buys detail with a much larger decode and a longer run,
	// so it stays an explicit request rather than the best-available default.
	int pt = pipeline_type;
	if( pt == T2_PIPE_AUTO ) {
		pt = p->cascade ? T2_PIPE_1024 : ( p->fine ? T2_PIPE_512 : T2_PIPE_COARSE );
	}
	if( is_cascade_pipeline( pt ) && !p->cascade )
		pt = p->fine ? T2_PIPE_512 : T2_PIPE_COARSE;
	if( pt == T2_PIPE_512 && !p->fine )
		pt = T2_PIPE_COARSE;

	// Intermediate shape-SLAT keyframes (opt-in via T2_KEYFRAMES = per-stage
	// count): capture a few x_0 estimates during the shape flow and replay them
	// as coarse meshes after the final decode. Only when the host takes previews.
	int keyframes = 0;
	if( preview ) {
		if( const char* kv = std::getenv( "T2_KEYFRAMES" ) )
			keyframes = std::atoi( kv );
		keyframes = keyframes < 0 ? 0 : ( keyframes > 8 ? 8 : keyframes );
	}

	const int S = 512;

	if( progress )
		progress( user, T2_STAGE_PREPROCESS, 0, 0 );
	std::vector<unsigned char> rgb( ( size_t )S * S * 3 );
	char					   perr[256] = { 0 };
	if( preprocess_image_bytes_mode( image_bytes, image_len, S, rgb.data(), background_mode, perr, sizeof( perr ) ) ) {
		copy_err( err, err_len, perr );
		return nullptr;
	}

	if( progress )
		progress( user, T2_STAGE_DINO, 0, 0 );
	trellis2_dino_cond cond; // 512-res conditioning (SS + LR flow)
	if( !trellis2_dino_encode_rgb( p->dino, rgb.data(), S, cond, &e ) ) {
		copy_err( err, err_len, "dino encode: " + e );
		return nullptr;
	}

	const trellis2_ss_flow_hparams& fhp = trellis2_ss_flow_hparams_of( p->flow );
	if( cond.channels() != fhp.cond_channels ) {
		copy_err( err, err_len, "cond/flow channel mismatch" );
		return nullptr;
	}

	// SS-decoder geometry (also the occupancy scratch reused by the live
	// previews and the settled decode below).
	const trellis2_ss_dec_hparams& dechp = trellis2_ss_dec_hparams_of( p->dec );
	const int					   Rout	 = dechp.res_out(); // 64
	std::vector<float>			   occ( ( size_t )dechp.out_channels * Rout * Rout * Rout );

	trellis2_ss_sampler_params	   sp;
	if( steps > 0 )
		sp.steps = steps;
	if( guidance >= 0 )
		sp.guidance_strength = guidance;
	sp.seed	   = seed;
	sp.verbose = false;
	struct cb_ctx {
		t2_progress_fn fn;
		void*		   user;
		int			   stage;
	} cbc { progress, user, T2_STAGE_SS_FLOW };
	if( progress ) {
		progress( user, T2_STAGE_SS_FLOW, 0, sp.steps );
		sp.progress = []( void* u, int step, int total ) {
			auto* c = ( cb_ctx* )u;
			c->fn( c->user, c->stage, step, total );
		};
		sp.progress_user = &cbc;
	}
	// Live per-step previews: decode each step's x_0 estimate into 64^3
	// occupancy and stream it as voxels (stride-gated; the last step is left to
	// the settled SS_DEC checkpoint below). `pctx`/`occ` outlive the sampler.
	ss_preview_ctx pctx;
	if( preview ) {
		pctx	   = ss_preview_ctx { preview, preview_user, p->dec, Rout, preview_stride( sp.steps ), &occ };
		sp.preview = []( void* u, int step, int total, const float* latent, int /*n*/ ) {
			auto* c = ( ss_preview_ctx* )u;
			if( step % c->stride != 0 || step == total )
				return;
			std::string de;
			if( !trellis2_ss_dec_decode( c->dec, latent, c->occ->data(), &de ) )
				return;
			std::vector<int32_t> cells;
			collect_occupied( c->occ->data(), c->res, cells );
			emit_voxels( c->fn, c->user, T2_STAGE_SS_FLOW, step, total, c->res, cells );
		};
		sp.preview_user = &pctx;
	}

	const int		   R = fhp.resolution;
	std::vector<float> latent( ( size_t )fhp.in_channels * R * R * R );
	if( !trellis2_ss_flow_sample( p->flow, cond.data.data(), ( int )cond.tokens(), ( int )cond.channels(), &sp, nullptr, latent.data(), &e ) ) {
		copy_err( err, err_len, "ss_flow sample: " + e );
		return nullptr;
	}

	if( progress )
		progress( user, T2_STAGE_SS_DEC, 0, 0 );
	if( !trellis2_ss_dec_decode( p->dec, latent.data(), occ.data(), &e ) ) {
		copy_err( err, err_len, "ss_dec decode: " + e );
		return nullptr;
	}
	// Settled-occupancy checkpoint (the clean sparse structure, 64^3 voxels).
	if( preview ) {
		std::vector<int32_t> cells;
		collect_occupied( occ.data(), Rout, cells );
		emit_voxels( preview, preview_user, T2_STAGE_SS_DEC, 0, 0, Rout, cells );
	}

	auto* r = new t2_mesh_result();

	if( pt == T2_PIPE_COARSE ) {
		// ── coarse path: marching cubes on the 64^3 occupancy ────────────────
		if( progress )
			progress( user, T2_STAGE_MESH, 0, 0 );
		mc::Mesh mesh = mc::extract( occ.data(), Rout, Rout, Rout, /*iso*/ 0.0f );
		if( mesh.verts.empty() ) {
			copy_err( err, err_len, "empty mesh (no occupied voxels at iso 0)" );
			delete r;
			return nullptr;
		}
		const float inv = 1.0f / ( float )Rout;
		for( size_t i = 0; i < mesh.verts.size(); ++i )
			mesh.verts[i] = mesh.verts[i] * inv - 0.5f;
		r->verts	= std::move( mesh.verts );
		r->normals	= std::move( mesh.normals );
		r->tris		= std::move( mesh.tris );
		r->grid_res = Rout; // the occupancy grid is the coarse path's resolution
		return r;
	}

	// ── fine / cascade: 64^3 occupancy -> 32^3 voxel scaffold ────────────────
	const trellis2_slat_flow_hparams& shp	 = trellis2_slat_flow_hparams_of( p->slat );
	const int						  ss_res = shp.resolution; // 32
	const int						  ratio	 = Rout / ss_res;  // 2 (max-pool 64 -> 32)
	std::vector<int32_t>			  coords;
	for( int x = 0; x < ss_res; ++x )
		for( int y = 0; y < ss_res; ++y )
			for( int z = 0; z < ss_res; ++z ) {
				bool any = false;
				for( int dx = 0; dx < ratio && !any; ++dx )
					for( int dy = 0; dy < ratio && !any; ++dy )
						for( int dz = 0; dz < ratio && !any; ++dz ) {
							const int	 xi = x * ratio + dx, yi = y * ratio + dy, zi = z * ratio + dz;
							const size_t idx = ( ( size_t )xi * Rout + yi ) * Rout + zi;
							if( occ[idx] > 0.0f )
								any = true;
						}
				if( any ) {
					coords.push_back( x );
					coords.push_back( y );
					coords.push_back( z );
				}
			}
	int L = ( int )( coords.size() / 3 );
	if( L == 0 ) {
		copy_err( err, err_len, "empty voxel scaffold" );
		delete r;
		return nullptr;
	}

	// shape-SLAT sampler params (shared LR + HR)
	auto make_slp = [&]( int stage, uint64_t sd ) {
		trellis2_ss_sampler_params slp;
		if( steps > 0 )
			slp.steps = steps;
		if( guidance >= 0 )
			slp.guidance_strength = guidance;
		slp.guidance_rescale = 0.5f;
		slp.rescale_t		 = 3.0f;
		slp.seed			 = sd;
		slp.verbose			 = false;
		if( progress ) {
			progress( user, stage, 0, slp.steps );
			slp.progress = []( void* u, int step, int total ) {
				auto* c = ( cb_ctx* )u;
				c->fn( c->user, c->stage, step, total );
			};
			cbc.stage		  = stage;
			slp.progress_user = &cbc;
		}
		return slp;
	};

	kf_capture		   kf_lr, kf_hr; // intermediate shape-flow keyframes (opt-in)

	// ── LR shape-SLAT flow (512 model, 512-res cond) ─────────────────────────
	std::vector<float> slat( ( size_t )L * shp.in_channels );
	{
		trellis2_ss_sampler_params slp = make_slp( T2_STAGE_SLAT_FLOW, seed ^ 0x51a7ULL );
		if( keyframes > 0 ) {
			kf_lr.stage		= T2_STAGE_SLAT_FLOW;
			kf_lr.res_in	= ss_res;
			kf_lr.channels	= shp.in_channels;
			kf_lr.norm_mean = shp.norm_mean;
			kf_lr.norm_std	= shp.norm_std;
			kf_lr.coords	= coords;
			while( ( ss_res << ( kf_lr.levels + 1 ) ) <= 128 )
				kf_lr.levels++; // -> 128^3
			if( kf_lr.levels < 1 )
				kf_lr.levels = 1;
			kf_lr.stride	 = std::max( 1, slp.steps / keyframes );
			slp.preview		 = kf_capture_cb;
			slp.preview_user = &kf_lr;
		}
		if( !trellis2_slat_flow_sample( p->slat, L, coords.data(), cond.data.data(), ( int )cond.tokens(), ( int )cond.channels(), &slp, nullptr, /*denormalize*/ true, slat.data(), &e ) ) {
			copy_err( err, err_len, "slat sample: " + e );
			delete r;
			return nullptr;
		}
	}

	const trellis2_shape_dec_hparams& dhp2 = trellis2_shape_dec_hparams_of( p->shapedec );
	std::vector<float>				  dec_feats; // 7-ch decoder output to mesh
	std::vector<int32_t>			  dec_coords;
	int								  grid = 0;
	trellis2_dino_cond				  cond1024; // filled by the cascade branch; reused by texturing

	if( pt == T2_PIPE_512 ) {
		// ── 512 fine: decode the LR slat directly at grid 512 ────────────────
		if( progress )
			progress( user, T2_STAGE_SHAPE_DEC, 0, 0 );
		ensure_decode_vram( p, T2_PIPE_512 ); // free the flow DiTs if a GPU decode needs the room
		if( !trellis2_shape_dec_decode( p->shapedec, slat.data(), L, coords.data(), dec_feats, dec_coords, nullptr, &e ) ) {
			copy_err( err, err_len, "shape decode: " + e );
			delete r;
			return nullptr;
		}
		grid = ss_res * dhp2.upscale(); // 32 * 16 = 512
	} else {
		// ── cascade: upsample -> quantize -> HR flow -> decode at 1024 or 1536 ─
		if( progress )
			progress( user, T2_STAGE_UPSAMPLE, 0, 0 );
		std::vector<int32_t> up_coords; // 512^3 candidate coords
		if( !trellis2_shape_dec_upsample( p->shapedec,
				slat.data(),
				L,
				coords.data(),
				/*upsample_times*/ 4,
				up_coords,
				&e ) ) {
			copy_err( err, err_len, "shape upsample: " + e );
			delete r;
			return nullptr;
		}
		// Quantize (c+0.5)/512*(res/16) and dedup into the HR scaffold, stepping
		// the requested resolution down by 128 while the scaffold would reach the
		// HR flow's token budget (t2cascade::select_scaffold). At T2_PIPE_1024
		// that loop quantizes once at 64^3 and stops on the floor, so this tier
		// stays exactly what it was.
		const int					 lr_res = ss_res * dhp2.upscale();		 // 512
		const int					 req_res = requested_cascade_resolution( pt ); // 1024 or 1536
		std::vector<int32_t>		 hr_coords;
		const t2cascade::selection	 sel	 = t2cascade::select_scaffold( up_coords, lr_res, req_res, cascade_max_tokens(), hr_coords );
		const int					 hr_grid = sel.grid; // 64 at 1024, 96 at 1536
		const int					 Lhr	 = sel.tokens;
		if( Lhr == 0 ) {
			copy_err( err, err_len, "empty HR scaffold" );
			delete r;
			return nullptr;
		}
		// Make the budget's decision visible instead of quietly returning a
		// coarser mesh than asked for — upstream prints a notice here.
		if( progress )
			progress( user, T2_STAGE_UPSAMPLE, sel.resolution, req_res );
		// The scaffold size is the one number that predicts the rest of the
		// run: the HR flow's cost is linear in it, and the decoder's per-level
		// voxel counts grow out of it. It is known here, minutes before either
		// can fail, so it is worth reporting rather than reconstructing later.
		if( std::getenv( "TRELLIS2_TIMING" ) )
			std::fprintf( stderr, "[cascade] requested %d, achieved %d (grid %d), %d scaffold tokens, %d reduction(s), budget %d\n", req_res, sel.resolution, hr_grid, Lhr, sel.reductions, cascade_max_tokens() );

		// Sharper HR-scaffold checkpoint (the cascade's refined structure).
		emit_voxels( preview, preview_user, T2_STAGE_UPSAMPLE, 0, 0, hr_grid, hr_coords );

		// 1024-res conditioning (separate preprocess + encode at 1024)
		std::vector<unsigned char> rgb1024( ( size_t )1024 * 1024 * 3 );
		if( preprocess_image_bytes_mode( image_bytes, image_len, 1024, rgb1024.data(), background_mode, perr, sizeof( perr ) ) ) {
			copy_err( err, err_len, perr );
			delete r;
			return nullptr;
		}
		if( !trellis2_dino_encode_rgb( p->dino, rgb1024.data(), 1024, cond1024, &e ) ) {
			copy_err( err, err_len, "dino encode 1024: " + e );
			delete r;
			return nullptr;
		}

		// HR shape-SLAT flow (1024 model)
		const trellis2_slat_flow_hparams& shp_hr = trellis2_slat_flow_hparams_of( p->slat_hr );
		std::vector<float>				  hr_slat( ( size_t )Lhr * shp_hr.in_channels );
		trellis2_ss_sampler_params		  slp = make_slp( T2_STAGE_SLAT_FLOW_HR, seed ^ 0x1024ULL );
		if( keyframes > 0 ) {
			kf_hr.stage		= T2_STAGE_SLAT_FLOW_HR;
			kf_hr.res_in	= hr_grid;
			kf_hr.channels	= shp_hr.in_channels;
			kf_hr.norm_mean = shp_hr.norm_mean;
			kf_hr.norm_std	= shp_hr.norm_std;
			kf_hr.coords	= hr_coords;
			while( ( hr_grid << ( kf_hr.levels + 1 ) ) <= 128 )
				kf_hr.levels++; // -> 128^3
			// One level is the minimum that produces a usable preview surface.
			// At hr_grid 64 that lands on 128^3; at 96 the first shift already
			// overshoots, so the keyframe grid is 192^3 (~28 MB of scratch per
			// frame). Opt-in (T2_KEYFRAMES) and off the generation path, so the
			// larger grid is a cost, not a correctness problem.
			if( kf_hr.levels < 1 )
				kf_hr.levels = 1;
			kf_hr.stride	 = std::max( 1, slp.steps / keyframes );
			slp.preview		 = kf_capture_cb;
			slp.preview_user = &kf_hr;
		}
		if( !trellis2_slat_flow_sample(
				p->slat_hr, Lhr, hr_coords.data(), cond1024.data.data(), ( int )cond1024.tokens(), ( int )cond1024.channels(), &slp, nullptr, /*denormalize*/ true, hr_slat.data(), &e ) ) {
			copy_err( err, err_len, "HR slat sample: " + e );
			delete r;
			return nullptr;
		}

		if( progress )
			progress( user, T2_STAGE_SHAPE_DEC_HR, 0, 0 );
		// Free the flow DiTs (all done) for the HR decode. Budget against what
		// the loop actually settled on, not what was requested: a 1536 request
		// that fell back to 1024 needs the 1024 threshold.
		ensure_decode_vram( p, sel.resolution > 1024 ? T2_PIPE_1536 : T2_PIPE_1024 );
		if( !trellis2_shape_dec_decode( p->shapedec, hr_slat.data(), Lhr, hr_coords.data(), dec_feats, dec_coords, nullptr, &e ) ) {
			copy_err( err, err_len, "HR shape decode: " + e );
			delete r;
			return nullptr;
		}
		grid = hr_grid * dhp2.upscale(); // 64 * 16 = 1024, 96 * 16 = 1536
	}

	// ── mesh extraction (shared) ─────────────────────────────────────────────
	r->grid_res = grid; // 512, or what the cascade token budget settled on
	if( progress )
		progress( user, T2_STAGE_MESH, 0, 0 );
	const int nvox = ( int )( dec_coords.size() / 3 );
	fdg::Mesh mesh = fdg::extract( dec_feats.data(), dec_coords.data(), nvox, grid );
	if( mesh.verts.empty() ) {
		copy_err( err, err_len, "empty mesh (dual grid found no faces)" );
		delete r;
		return nullptr;
	}
	// Clean up the raw dual-grid soup: drop floating specks, then fill the small
	// boundary holes extract() left. Both only edit triangles (no new vertices),
	// so the generated material volume can still be sampled at every vertex.
	fdg::drop_small_components( mesh );
	// fill_holes' loop limit counts boundary EDGES, so it is not normalised
	// against the grid: the same physical opening has proportionally more edges
	// at a finer grid, and stops being filled somewhere between tiers. That is
	// the mechanism behind holes appearing at 1024 and 1536 that 512 did not
	// have, on the same object and seed. Scaling the limit with the grid keeps
	// the threshold at a fixed physical size, which is what it was meant to be.
	// (drop_small_components above is already relative, so it needs nothing.)
	//
	// The cap exists because a loop is fan-triangulated from its first vertex,
	// which degrades on large non-planar boundaries — so this raises the size
	// of opening that gets a rough fill, it does not make filling free.
	fdg::fill_holes( mesh, std::max( 64, 64 * grid / 512 ) );

	// What fill_holes could not close, as a number rather than an impression.
	// A boundary edge belongs to exactly one triangle; on a closed surface there
	// are none. Reporting the count and the largest loop separates "the limit is
	// still too low" from "the extractor left genuine openings" — the two need
	// opposite fixes, and a wireframe screenshot at 3M triangles cannot tell them
	// apart.
	if( std::getenv( "TRELLIS2_TIMING" ) ) {
		std::unordered_map<uint64_t, int> edge_use;
		edge_use.reserve( mesh.tris.size() );
		auto ekey = []( int a, int b ) {
			const uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
			return ( ( uint64_t )lo << 32 ) | hi;
		};
		for( size_t t = 0; t + 2 < mesh.tris.size(); t += 3 ) {
			edge_use[ekey( mesh.tris[t], mesh.tris[t + 1] )]++;
			edge_use[ekey( mesh.tris[t + 1], mesh.tris[t + 2] )]++;
			edge_use[ekey( mesh.tris[t + 2], mesh.tris[t] )]++;
		}
		size_t				 boundary = 0, nonmanifold = 0;
		std::vector<int32_t> bverts;
		for( const auto& e : edge_use ) {
			if( e.second == 1 ) {
				++boundary;
				bverts.push_back( ( int32_t )( e.first >> 32 ) );
				bverts.push_back( ( int32_t )( e.first & 0xffffffffu ) );
			} else if( e.second > 2 )
				++nonmanifold;
		}
		std::fprintf( stderr,
			"[mesh] grid %d: %zu verts, %zu tris, %zu boundary edges (%.4f%% of %zu), %zu non-manifold edges, fill limit %d\n",
			grid,
			mesh.verts.size() / 3,
			mesh.tris.size() / 3,
			boundary,
			edge_use.empty() ? 0.0 : 100.0 * ( double )boundary / ( double )edge_use.size(),
			edge_use.size(),
			nonmanifold,
			std::max( 64, 64 * grid / 512 ) );
	}
	r->normals = fdg::vertex_normals( mesh );
	// The dual grid is unoriented. Our viewer shades around that, but glTF
	// consumers and the wrap/remesh stages read the winding, so settle it here —
	// last, once the triangle set is final and the normals exist to orient against.
	fdg::orient_faces( mesh, r->normals );
	r->verts = std::move( mesh.verts );
	r->tris	 = std::move( mesh.tris );

	// ── shape-flow keyframe replay ───────────────────────────────────────────
	// Now is the safe window: the flow DiTs are freed (ensure_decode_vram) and
	// the shape decoder owns VRAM, before the texture stage loads its models.
	if( preview && keyframes > 0 ) {
		emit_keyframes( p->shapedec, kf_lr, preview, preview_user );
		emit_keyframes( p->shapedec, kf_hr, preview, preview_user ); // empty for 512
	}

	// ── PBR texture stage (optional) ─────────────────────────────────────────
	if( p->texture && pt != T2_PIPE_COARSE ) {
		// Free the (finished) geometry flow DiTs so the ~4 GB of tex models fit
		// in VRAM; reload_flows() restores them on the next generate.
		if( trellis2_gpu_free_vram() > 0 ) {
			if( p->flow ) {
				trellis2_ss_flow_free( p->flow );
				p->flow = nullptr;
			}
			if( p->slat ) {
				trellis2_slat_flow_free( p->slat );
				p->slat = nullptr;
			}
			if( p->slat_hr ) {
				trellis2_slat_flow_free( p->slat_hr );
				p->slat_hr = nullptr;
			}
		}
		const trellis2_dino_cond& texcond = is_cascade_pipeline( pt ) ? cond1024 : cond;
		std::vector<float>		  pbr;
		std::string				  te;
		if( !run_texture_stage( p, dec_feats, dec_coords, r->verts, grid, pt, texcond, seed ^ 0x7ec0ULL, texture_steps, progress, user, pbr, te ) ) {
			copy_err( err, err_len, "texture: " + te );
			delete r;
			return nullptr;
		}
		r->pbr = std::move( pbr );
	}
	return r;
}

int t2_mesh_n_verts( const t2_mesh_result* r )
{
	return r ? ( int )( r->verts.size() / 3 ) : 0;
}
int t2_mesh_n_tris( const t2_mesh_result* r )
{
	return r ? ( int )( r->tris.size() / 3 ) : 0;
}
const float* t2_mesh_verts( const t2_mesh_result* r )
{
	return r ? r->verts.data() : nullptr;
}
const float* t2_mesh_normals( const t2_mesh_result* r )
{
	return r ? r->normals.data() : nullptr;
}
const int* t2_mesh_tris( const t2_mesh_result* r )
{
	return r ? r->tris.data() : nullptr;
}
int t2_mesh_has_pbr( const t2_mesh_result* r )
{
	return ( r && !r->pbr.empty() ) ? 1 : 0;
}
const float* t2_mesh_pbr( const t2_mesh_result* r )
{
	return ( r && !r->pbr.empty() ) ? r->pbr.data() : nullptr;
}
int t2_mesh_grid_resolution( const t2_mesh_result* r )
{
	return r ? r->grid_res : 0;
}
void t2_mesh_free( t2_mesh_result* r )
{
	delete r;
}

t2_mesh_result* t2_prepare_mesh( const float* verts, int n_verts, const int* tris, int n_tris, const float* pbr, int component_filter, char* err, int err_len )
{
	if( !verts || !tris || n_verts <= 0 || n_tris <= 0 ) {
		copy_err( err, err_len, "empty mesh" );
		return nullptr;
	}
	if( component_filter < 0 || component_filter > 2 ) {
		copy_err( err, err_len, "bad component filter" );
		return nullptr;
	}
	t2glb::MeshExportOptions opt;
	opt.components = ( t2glb::ComponentFilter )component_filter;
	t2glb::PreparedMesh prepared;
	std::string			e;
	if( !t2glb::prepare_mesh( verts, n_verts, ( const int32_t* )tris, n_tris, pbr, opt, prepared, e ) ) {
		copy_err( err, err_len, e );
		return nullptr;
	}
	auto* r	   = new t2_mesh_result();
	r->verts   = std::move( prepared.verts );
	r->normals = std::move( prepared.normals );
	r->tris.assign( prepared.tris.begin(), prepared.tris.end() );
	r->pbr = std::move( prepared.pbr );
	return r;
}

int t2_print_remesh_available()
{
	return t2glb::print_remesh_available() ? 1 : 0;
}

t2_mesh_result* t2_prepare_print_mesh(
	const float* verts, int n_verts, const int* tris, int n_tris, const float* pbr, int component_filter, float alpha_ratio, float offset_ratio, char* err, int err_len )
{
	if( !verts || !tris || n_verts <= 0 || n_tris <= 0 ) {
		copy_err( err, err_len, "empty mesh" );
		return nullptr;
	}
	if( component_filter < 0 || component_filter > 2 ) {
		copy_err( err, err_len, "bad component filter" );
		return nullptr;
	}
	t2glb::MeshExportOptions opt;
	opt.components = ( t2glb::ComponentFilter )component_filter;
	t2glb::PreparedMesh prepared;
	std::string			e;
	if( !t2glb::prepare_print_mesh( verts, n_verts, ( const int32_t* )tris, n_tris, pbr, opt, alpha_ratio, offset_ratio, prepared, e ) ) {
		copy_err( err, err_len, e );
		return nullptr;
	}
	auto* r	   = new t2_mesh_result();
	r->verts   = std::move( prepared.verts );
	r->normals = std::move( prepared.normals );
	r->tris.assign( prepared.tris.begin(), prepared.tris.end() );
	r->pbr = std::move( prepared.pbr );
	return r;
}

const char* t2_projection_backend()
{
	return t2print::projection_backend();
}

void t2_last_bake_timings( float* out_unwrap, float* out_rasterize, float* out_projection, float* out_texel_fill, float* out_encode )
{
	const t2glb::BakeTimings t = t2glb::last_bake_timings();
	if( out_unwrap )
		*out_unwrap = ( float )t.unwrap;
	if( out_rasterize )
		*out_rasterize = ( float )t.rasterize;
	if( out_projection )
		*out_projection = ( float )t.projection;
	if( out_texel_fill )
		*out_texel_fill = ( float )t.texel_fill;
	if( out_encode )
		*out_encode = ( float )t.encode;
}

int t2_quad_remesh_available()
{
	return t2glb::quad_remesh_available() ? 1 : 0;
}

t2_mesh_result* t2_prepare_quad_mesh( const float* verts, int n_verts, const int* tris, int n_tris, const float* pbr, int component_filter, int target_quads, float adaptivity, char* err, int err_len )
{
	if( !verts || !tris || n_verts <= 0 || n_tris <= 0 ) {
		copy_err( err, err_len, "empty mesh" );
		return nullptr;
	}
	if( component_filter < 0 || component_filter > 2 ) {
		copy_err( err, err_len, "bad component filter" );
		return nullptr;
	}
	t2glb::MeshExportOptions opt;
	opt.components = ( t2glb::ComponentFilter )component_filter;
	t2glb::PreparedMesh	 prepared;
	t2glb::QuadMeshStats stats;
	std::string			 e;
	if( !t2glb::prepare_quad_mesh( verts, n_verts, ( const int32_t* )tris, n_tris, pbr, opt, target_quads, adaptivity, prepared, &stats, e ) ) {
		copy_err( err, err_len, e );
		return nullptr;
	}
	auto* r	   = new t2_mesh_result();
	r->verts   = std::move( prepared.verts );
	r->normals = std::move( prepared.normals );
	r->tris.assign( prepared.tris.begin(), prepared.tris.end() );
	r->pbr			  = std::move( prepared.pbr );
	r->quad_stats	  = stats;
	r->has_quad_stats = true;
	return r;
}

void t2_quad_mesh_stats( const t2_mesh_result* r, int* out_quads, int* out_triangles, int* out_ngons, int* out_boundary_edges, float* out_area_retained )
{
	const bool ok = r && r->has_quad_stats;
	if( out_quads )
		*out_quads = ok ? r->quad_stats.quads : 0;
	if( out_triangles )
		*out_triangles = ok ? r->quad_stats.triangles : 0;
	if( out_ngons )
		*out_ngons = ok ? r->quad_stats.ngons : 0;
	if( out_boundary_edges )
		*out_boundary_edges = ok ? r->quad_stats.boundary_edges : 0;
	if( out_area_retained )
		*out_area_retained = ok ? r->quad_stats.area_retained : 0.0f;
}

uint8_t* t2_bake_glb( const float* verts, int n_verts, const int* tris, int n_tris, const float* pbr, int texture_size, int component_filter, int* out_len, char* err, int err_len )
{
	if( out_len )
		*out_len = 0;
	if( !verts || !tris || n_verts <= 0 || n_tris <= 0 ) {
		copy_err( err, err_len, "empty mesh" );
		return nullptr;
	}
	t2glb::MeshExportOptions opt;
	if( texture_size > 0 )
		opt.texture_size = texture_size;
	if( component_filter < 0 || component_filter > 2 ) {
		copy_err( err, err_len, "bad component filter" );
		return nullptr;
	}
	opt.components = ( t2glb::ComponentFilter )component_filter;
	std::vector<uint8_t> glb;
	std::string			 e;
	if( !t2glb::mesh_to_glb( verts, n_verts, tris, n_tris, pbr, opt, glb, e ) ) {
		copy_err( err, err_len, e );
		return nullptr;
	}
	uint8_t* buf = ( uint8_t* )std::malloc( glb.size() );
	if( !buf ) {
		copy_err( err, err_len, "out of memory" );
		return nullptr;
	}
	std::memcpy( buf, glb.data(), glb.size() );
	if( out_len )
		*out_len = ( int )glb.size();
	return buf;
}

uint8_t* t2_bake_projected_glb( const float* target_verts,
	int										 target_n_verts,
	const int*								 target_tris,
	int										 target_n_tris,
	const float*							 source_verts,
	int										 source_n_verts,
	const int*								 source_tris,
	int										 source_n_tris,
	const float*							 source_pbr,
	int										 texture_size,
	int										 source_component_filter,
	int										 normal_map,
	int*									 out_len,
	char*									 err,
	int										 err_len )
{
	if( out_len )
		*out_len = 0;
	if( !target_verts || !target_tris || target_n_verts <= 0 || target_n_tris <= 0 || !source_verts || !source_tris || !source_pbr || source_n_verts <= 0 || source_n_tris <= 0 ) {
		copy_err( err, err_len, "empty projected GLB mesh" );
		return nullptr;
	}
	if( source_component_filter < 0 || source_component_filter > 2 ) {
		copy_err( err, err_len, "bad component filter" );
		return nullptr;
	}
	t2glb::MeshExportOptions opt;
	if( texture_size > 0 )
		opt.texture_size = texture_size;
	opt.components = ( t2glb::ComponentFilter )source_component_filter;
	opt.normal_map = normal_map != 0;
	std::vector<uint8_t> glb;
	std::string			 e;
	if( !t2glb::mesh_to_projected_glb(
			target_verts, target_n_verts, ( const int32_t* )target_tris, target_n_tris, source_verts, source_n_verts, ( const int32_t* )source_tris, source_n_tris, source_pbr, opt, glb, e ) ) {
		copy_err( err, err_len, e );
		return nullptr;
	}
	uint8_t* buf = ( uint8_t* )std::malloc( glb.size() );
	if( !buf ) {
		copy_err( err, err_len, "out of memory" );
		return nullptr;
	}
	std::memcpy( buf, glb.data(), glb.size() );
	if( out_len )
		*out_len = ( int )glb.size();
	return buf;
}

void t2_last_normal_map_stats( int* out_covered_texels, int* out_rejected_texels )
{
	const t2glb::BakeTimings t = t2glb::last_bake_timings();
	if( out_covered_texels )
		*out_covered_texels = t.normal_texels;
	if( out_rejected_texels )
		*out_rejected_texels = t.normal_rejected;
}

void t2_free_buffer( uint8_t* buf )
{
	std::free( buf );
}

} // extern "C"
