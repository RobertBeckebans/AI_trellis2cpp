#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/*
** ── DLL Export Decoration ────────────────────────────────────────────────
**
** TRELLIS2_API mirrors LLAMA_API / SD_API / SAM3_API: tag every public
** function so MSVC emits the correct __declspec when trellis2 is built /
** consumed as a shared library. For static builds the macro expands to
** nothing and there is no ABI surface.
*/
#ifdef TRELLIS2_SHARED
	#if defined( _WIN32 ) && !defined( __MINGW32__ )
		#ifdef TRELLIS2_BUILD
			#define TRELLIS2_API __declspec( dllexport )
		#else
			#define TRELLIS2_API __declspec( dllimport )
		#endif
	#else
		#define TRELLIS2_API __attribute__( ( visibility( "default" ) ) )
	#endif
#else
	#define TRELLIS2_API
#endif

/*
** ── Version ─────────────────────────────────────────────────────────────
*/

#define TRELLIS2_VERSION_MAJOR 0
#define TRELLIS2_VERSION_MINOR 1
#define TRELLIS2_VERSION_PATCH 0
#define TRELLIS2_VERSION	   "0.1.0"

/*****************************************************************************
** Public Data Types
*****************************************************************************/

/*
** ── DINOv3 conditioning tensor ───────────────────────────────────────────
**
** The exact image-conditioning that TRELLIS.2 stage 1 (sparse-structure flow
** DiT) consumes as cross-attention K/V. It is the full DINOv3 ViT-L/16 token
** sequence of the LAST transformer layer with an AFFINE-FREE LayerNorm on top
** (NOT HF's learned-final-norm `last_hidden_state`).
**
** Canonical shape is [1, N, C] = [1, 1029, 1024], where the 1029 tokens are
** 1 CLS + 4 register + 1024 patch (32x32 @ 512px, patch-16). Cross-attention
** is permutation-invariant over these tokens (no positional embedding on the
** cond side), so token order is irrelevant — but the full set must be present.
**
** neg_cond (for classifier-free guidance) is exactly zeros_like(cond) and is
** therefore never stored on disk.
*/
struct trellis2_dino_cond {
	// Row-major (C-contiguous) shape. ndim is normally 3 -> {1, N, C}.
	std::vector<int64_t> shape;
	// prod(shape) float32 values, C-contiguous in `shape` order.
	std::vector<float>	 data;

	uint32_t			 format_version = 0; // version field from the .dinodata header

	int64_t				 ndim() const { return ( int64_t )shape.size(); }
	int64_t				 batch() const { return shape.size() == 3 ? shape[0] : 1; }
	int64_t				 tokens() const { return shape.size() == 3 ? shape[1] : ( shape.empty() ? 0 : shape[0] ); }
	int64_t				 channels() const { return shape.empty() ? 0 : shape.back(); }
	size_t				 count() const { return data.size(); }
	bool				 empty() const { return data.empty(); }
};

/*
** Cheap fingerprints over the whole payload — these mirror exactly the values
** written into the `<stem>.dino.txt` JSON sidecars by the Python side, so they
** can be used to verify that this loader read the file bit-for-bit, and later
** that an ONNX-derived cond matches the reference.
*/
struct trellis2_dino_fingerprint {
	float  vmin	 = 0.0f;
	float  vmax	 = 0.0f;
	double mean	 = 0.0;
	double sum	 = 0.0;
	double l2	 = 0.0;
	size_t count = 0;
};

/*
** ── Sparse-structure flow DiT (stage 1) ──────────────────────────────────
**
** Hyperparameters of the SparseStructureFlowModel, read from the GGUF KV
** store (`trellis2.ss_flow.*`). For the shipped 1.3B checkpoint these are:
** resolution 16, in/out 8, model_channels 1536, cond_channels 1024,
** num_blocks 30, num_heads 12, mlp_ratio 5.3334, rope PE, share_mod + QK-RMS
** norm on both self- and cross-attention.
*/
struct trellis2_ss_flow_hparams {
	int32_t resolution		  = 0;
	int32_t in_channels		  = 0;
	int32_t out_channels	  = 0;
	int32_t model_channels	  = 0;
	int32_t cond_channels	  = 0;
	int32_t num_blocks		  = 0;
	int32_t num_heads		  = 0;
	float	mlp_ratio		  = 0.0f;
	int32_t share_mod		  = 0; // bool
	int32_t qk_rms_norm		  = 0; // bool
	int32_t qk_rms_norm_cross = 0; // bool
	float	rope_freq_min	  = 1.0f;
	float	rope_freq_base	  = 10000.0f;
	char	pe_mode[16]		  = { 0 }; // "rope" or "ape"
	int32_t file_type		  = 0;	   // 0=f32, 1=f16, 2=bf16

	int32_t head_dim() const { return num_heads ? model_channels / num_heads : 0; }
};

// Opaque handle to a loaded SS-flow model (weights + metadata).
struct trellis2_ss_flow_model;

// Lightweight description of one weight tensor, for inspection.
struct trellis2_tensor_info {
	std::string name;
	int			n_dims	  = 0;
	int64_t		ne[4]	  = { 1, 1, 1, 1 }; // ggml order (ne[0] is fastest-moving)
	int			ggml_type = 0;
	std::string type_name;
	size_t		n_bytes = 0;
};

/*****************************************************************************
** Public API – Version
*****************************************************************************/

TRELLIS2_API const char*					 trellis2_version();

/*****************************************************************************
** Public API – Sparse-structure flow DiT (stage 1)
*****************************************************************************/

// Load the SS-flow DiT from a GGUF produced by convert_ss_flow_to_gguf.py.
// If `load_tensors` is true the weight data is read into host memory; if false
// only the metadata + tensor descriptors are parsed (fast, for inspection).
// Returns nullptr on failure and (if `error`) fills it with a reason.
TRELLIS2_API trellis2_ss_flow_model*		 trellis2_ss_flow_load( const std::string& path, bool load_tensors = true, std::string* error = nullptr, const char* device = nullptr );

TRELLIS2_API void							 trellis2_ss_flow_free( trellis2_ss_flow_model* m );

// Name of the compute backend the model was loaded onto (e.g. the GPU device
// description, or "CPU"). Returns "none" if loaded metadata-only.
TRELLIS2_API const char*					 trellis2_ss_flow_backend_name( const trellis2_ss_flow_model* m );

TRELLIS2_API const trellis2_ss_flow_hparams& trellis2_ss_flow_hparams_of( const trellis2_ss_flow_model* m );

TRELLIS2_API int							 trellis2_ss_flow_n_tensors( const trellis2_ss_flow_model* m );
TRELLIS2_API bool							 trellis2_ss_flow_get_tensor_info( const trellis2_ss_flow_model* m, int i, trellis2_tensor_info& out );
// True iff a tensor with this exact name is present.
TRELLIS2_API bool							 trellis2_ss_flow_has_tensor( const trellis2_ss_flow_model* m, const std::string& name );

// Flow-Euler sampler parameters for the sparse-structure stage. Defaults match
// the microsoft/TRELLIS.2-4B pipeline.json (FlowEulerGuidanceIntervalSampler).
struct trellis2_ss_sampler_params {
	int		 steps				   = 12;
	float	 guidance_strength	   = 7.5f;
	float	 guidance_rescale	   = 0.7f;
	float	 guidance_interval_min = 0.6f;
	float	 guidance_interval_max = 1.0f;
	float	 rescale_t			   = 5.0f;
	float	 sigma_min			   = 1e-5f;
	uint64_t seed				   = 0;	   // used only when noise is generated internally
	bool	 verbose			   = true; // print per-step progress to stderr

	// Optional per-step progress callback (called after each Euler step).
	void ( *progress )( void* user, int step, int total ) = nullptr;
	void* progress_user									  = nullptr;

	// Optional per-step preview callback: receives the denoised x_0 estimate
	// (the current best-guess clean latent, [in_channels * resolution^3]) after
	// each step, so a host can decode it into a live intermediate 3D preview.
	// `latent` is valid only during the call. Runs on the sampling thread — the
	// callee must return promptly (it blocks the next step).
	void ( *preview )( void* user, int step, int total, const float* latent, int n ) = nullptr;
	void* preview_user																 = nullptr;
};

// Run one forward pass of the SS-flow DiT (CPU backend), i.e. the velocity
// prediction v = model(x, t, cond). The model must have been loaded with
// load_tensors=true.
//
//   x    : [in_channels * resolution^3] floats, channel-major
//          (x[c*R^3 + n], n = i*R^2 + j*R + k over the R^3 grid) — exactly the
//          flattened latent the Python forward sees as [1, C, R, R, R].
//   t    : scalar timestep, fed verbatim to the sinusoidal timestep embedding.
//   cond : [cond_tokens * cond_channels] floats, token-major (the .dinodata
//          layout). Pass all-zero data for the CFG negative branch.
//   out  : [out_channels * resolution^3] floats, channel-major (same as x).
//
// Returns false on failure and (if `error`) fills it with a reason.
TRELLIS2_API bool trellis2_ss_flow_forward( trellis2_ss_flow_model* m, const float* x, float t, const float* cond, int cond_tokens, int cond_channels, float* out, std::string* error = nullptr );

// Run the full flow-Euler sampling loop (classifier-free guidance with interval
// + rescale) to produce the stage-1 sparse-structure latent z_s.
//
//   cond       : [cond_tokens * cond_channels] DINOv3 tokens (the .dinodata
//                layout). The negative branch uses zeros_like(cond) internally.
//   params     : sampler settings; nullptr uses the pipeline defaults above.
//   noise      : optional [in_channels * resolution^3] initial noise
//                (channel-major). If nullptr, standard-normal noise is generated
//                from params->seed.
//   out_latent : [in_channels * resolution^3] floats, channel-major — the z_s
//                latent to feed into the SS decoder.
//
// Returns false on failure and (if `error`) fills it with a reason.
TRELLIS2_API bool trellis2_ss_flow_sample(
	trellis2_ss_flow_model* m, const float* cond, int cond_tokens, int cond_channels, const trellis2_ss_sampler_params* params, const float* noise, float* out_latent, std::string* error = nullptr );

/*****************************************************************************
** Public API – Sparse-structure decoder (stage 1)
**
** D_S in the paper: a dense 3D-conv ResNet (SparseStructureDecoder) that turns
** the sparse-structure latent z_s ([latent_channels, R_in^3]) into an occupancy
** logit grid ([out_channels, R_out^3]). For the shipped ss_dec_conv3d_16l8
** checkpoint: latent_channels 8, out_channels 1, channels [512,128,32],
** num_res_blocks 2, num_res_blocks_middle 2, "layer" (channel-LayerNorm) norm.
** Two pixel-shuffle upsamples take R_in=16 -> 32 -> R_out=64. The coarse voxel
** scaffold is `logit > 0`.
*****************************************************************************/

struct trellis2_ss_dec_hparams {
	int32_t out_channels		  = 0;
	int32_t latent_channels		  = 0;
	int32_t num_res_blocks		  = 0;
	int32_t num_res_blocks_middle = 0;
	int32_t n_levels			  = 0;
	int32_t channels[8]			  = { 0 };
	float	norm_eps			  = 1e-5f;
	int32_t file_type			  = 0;	   // 0=f32, 1=f16
	char	norm_type[16]		  = { 0 }; // "layer"

	// The decoder upsamples by 2 per level transition (n_levels-1 of them).
	int32_t res_in() const { return 16; }
	int32_t upscale() const
	{
		int u = 1;
		for( int i = 1; i < n_levels; ++i )
			u *= 2;
		return u;
	}
	int32_t res_out() const { return res_in() * upscale(); }
};

// Opaque handle to a loaded SS decoder (weights + metadata).
struct trellis2_ss_dec_model;

// Load the SS decoder from a GGUF produced by convert_ss_dec_to_gguf.py.
TRELLIS2_API trellis2_ss_dec_model*			trellis2_ss_dec_load( const std::string& path, bool load_tensors = true, std::string* error = nullptr, const char* device = nullptr );

TRELLIS2_API void							trellis2_ss_dec_free( trellis2_ss_dec_model* m );

TRELLIS2_API const char*					trellis2_ss_dec_backend_name( const trellis2_ss_dec_model* m );

TRELLIS2_API const trellis2_ss_dec_hparams& trellis2_ss_dec_hparams_of( const trellis2_ss_dec_model* m );

TRELLIS2_API int							trellis2_ss_dec_n_tensors( const trellis2_ss_dec_model* m );
TRELLIS2_API bool							trellis2_ss_dec_get_tensor_info( const trellis2_ss_dec_model* m, int i, trellis2_tensor_info& out );

// Decode a sparse-structure latent z_s into an occupancy logit grid.
//
//   latent : [latent_channels * res_in^3] floats, channel-major
//            (latent[c*R^3 + n], n = i*R^2 + j*R + k) — exactly the z_s that
//            trellis2_ss_flow_sample() produces ([1, C, R, R, R] flattened).
//   out    : [out_channels * res_out^3] floats, channel-major. For the shipped
//            checkpoint out_channels=1 and res_out=64, so this is a 64^3 grid of
//            occupancy logits; the voxel scaffold is `out[n] > 0`.
//
// Returns false on failure and (if `error`) fills it with a reason.
TRELLIS2_API bool							trellis2_ss_dec_decode( trellis2_ss_dec_model* m, const float* latent, float* out, std::string* error = nullptr );

/*****************************************************************************
** Public API – Shape-SLAT flow DiT (stage 2, fine geometry)
**
** SLatFlowModel: the same 1.3B DiT block structure as the SS-flow model, but
** sparse — tokens are the active voxels of the 32^3 scaffold produced by the
** SS stage, with 3D RoPE over each voxel's integer coordinates. In/out are
** 32-channel per-voxel structured latents.
*****************************************************************************/

struct trellis2_slat_flow_hparams {
	int32_t resolution		  = 0; // 32 for the 512 model
	int32_t in_channels		  = 0; // 32
	int32_t out_channels	  = 0; // 32
	int32_t model_channels	  = 0; // 1536
	int32_t cond_channels	  = 0; // 1024
	int32_t num_blocks		  = 0; // 30
	int32_t num_heads		  = 0; // 12
	float	mlp_ratio		  = 0.0f;
	int32_t share_mod		  = 0;
	int32_t qk_rms_norm		  = 0;
	int32_t qk_rms_norm_cross = 0;
	float	rope_freq_min	  = 1.0f;
	float	rope_freq_base	  = 10000.0f;
	char	pe_mode[16]		  = { 0 };
	int32_t file_type		  = 0;

	// shape_slat_normalization (pipeline.json), baked in by the converter:
	// decoder input = sampler output * std + mean.
	float	norm_mean[64] = { 0 };
	float	norm_std[64]  = { 0 };

	// Texture SLAT flow only (concat_cond). 0 on the shape flow. When > 0 the DiT
	// is conditioned on the shape via concatenation: in_channels = out_channels +
	// concat_cond_channels (64 = 32 tex-noise + 32 shape-SLAT). The shape SLAT is
	// normalized by concat_norm_* before being concatenated onto the noise.
	int32_t concat_cond_channels = 0;
	float	concat_norm_mean[64] = { 0 };
	float	concat_norm_std[64]	 = { 0 };

	int32_t head_dim() const { return num_heads ? model_channels / num_heads : 0; }
};

struct trellis2_slat_flow_model;

TRELLIS2_API trellis2_slat_flow_model*		   trellis2_slat_flow_load( const std::string& path, bool load_tensors = true, std::string* error = nullptr, const char* device = nullptr );

TRELLIS2_API void							   trellis2_slat_flow_free( trellis2_slat_flow_model* m );
TRELLIS2_API const char*					   trellis2_slat_flow_backend_name( const trellis2_slat_flow_model* m );
TRELLIS2_API const trellis2_slat_flow_hparams& trellis2_slat_flow_hparams_of( const trellis2_slat_flow_model* m );

// One forward pass v = model(x, t, cond) over L active voxels.
//   x      : [L * in_channels] floats, voxel-major (x[v*C + c])
//   coords : [L * 3] int32 voxel coordinates (c1, c2, c3) in [0, resolution)
//   cond   : [cond_tokens * cond_channels], token-major (.dinodata layout)
//   out    : [L * out_channels], voxel-major
TRELLIS2_API bool							   trellis2_slat_flow_forward(
								 trellis2_slat_flow_model* m, const float* x, int n_voxels, const int32_t* coords, float t, const float* cond, int cond_tokens, int cond_channels, float* out, std::string* error = nullptr );

// Full flow-Euler sampling (CFG + interval + rescale) over the voxel set.
// Defaults for the shape stage differ from the SS stage: guidance_rescale 0.5,
// rescale_t 3.0 (set them in `params`; steps 12 / strength 7.5 are shared).
// When denormalize is true the output is sampler_output * std + mean, ready
// for the shape decoder.
TRELLIS2_API bool trellis2_slat_flow_sample( trellis2_slat_flow_model* m,
	int																   n_voxels,
	const int32_t*													   coords,
	const float*													   cond,
	int																   cond_tokens,
	int																   cond_channels,
	const trellis2_ss_sampler_params*								   params,
	const float*													   noise,
	bool															   denormalize,
	float*															   out_latent,
	std::string*													   error = nullptr );

// Texture-SLAT flow sampling with concat_cond (requires concat_cond_channels>0).
// Diffuses out_channels (32) and, each step, concatenates the shape SLAT
// (normalized by concat_norm_*) onto the noise so the DiT sees in_channels (64).
//   shape_slat : [L * concat_cond_channels] the (un-normalized) shape SLAT.
//   noise      : [L * out_channels] initial noise, or null to seed from params.
//   out_latent : [L * out_channels]; when denormalize, * norm_std + norm_mean.
TRELLIS2_API bool trellis2_slat_flow_sample_tex( trellis2_slat_flow_model* m,
	int																	   n_voxels,
	const int32_t*														   coords,
	const float*														   cond,
	int																	   cond_tokens,
	int																	   cond_channels,
	const float*														   shape_slat,
	const trellis2_ss_sampler_params*									   params,
	const float*														   noise,
	bool																   denormalize,
	float*																   out_latent,
	std::string*														   error = nullptr );

/*****************************************************************************
** Public API – Shape-SLAT VAE decoder (FlexiDualGridVaeDecoder)
**
** Sparse ConvNeXt U-Net decoder: 32-channel latent on the 32^3 voxel set ->
** 7 channels per active voxel at 16x resolution (512^3). The subdivision at
** each of the 4 upsampling steps is predicted by the network, so the active
** set grows adaptively. Output channels: [0:3] dual-vertex offset logits
** (sigmoid+margin applied by the mesher), [3:6] per-axis intersection logits,
** [6] quad split weight (softplus applied by the mesher).
*****************************************************************************/

struct trellis2_shape_dec_hparams {
	int32_t latent_channels = 0;	 // 32
	int32_t out_channels	= 0;	 // 7
	int32_t n_levels		= 0;	 // 5
	int32_t channels[8]		= { 0 }; // [1024, 512, 256, 128, 64]
	int32_t num_blocks[8]	= { 0 }; // [4, 16, 8, 4, 0]
	float	norm_eps		= 1e-6f;
	float	voxel_margin	= 0.5f;
	int32_t file_type		= 0;

	int32_t upscale() const
	{
		int u = 1;
		for( int i = 1; i < n_levels; ++i )
			u *= 2;
		return u;
	}
};

struct trellis2_shape_dec_model;

TRELLIS2_API trellis2_shape_dec_model*		   trellis2_shape_dec_load( const std::string& path, bool load_tensors = true, std::string* error = nullptr, const char* device = nullptr );

TRELLIS2_API void							   trellis2_shape_dec_free( trellis2_shape_dec_model* m );
TRELLIS2_API const char*					   trellis2_shape_dec_backend_name( const trellis2_shape_dec_model* m );
TRELLIS2_API const trellis2_shape_dec_hparams& trellis2_shape_dec_hparams_of( const trellis2_shape_dec_model* m );

// Optional per-level activation taps for validation (names match
// scripts/dump_slat_reference.py: "lvl{i}.pre_up", "lvl{i}.subdiv",
// "lvl{i}.in_coords", "out7", "out_coords").
struct trellis2_shape_dec_taps {
	std::vector<std::string>		names;
	std::vector<std::vector<float>> data;
};

// One decoder upsample level. For every fine (output) voxel, cidx identifies
// the selected child in the coarse [C*8, L_coarse] feature tensor. Shape
// decoding can return these levels so another decoder (notably the texture VAE)
// can reproduce exactly the same sparse hierarchy.
struct trellis2_subdiv_level {
	std::vector<int32_t> fine_coords; // [Lf * 3]
	std::vector<int32_t> cidx;		  // [Lf]
};

// Decode the (denormalized) structured latent into per-voxel dual-grid fields.
//   slat       : [L * latent_channels] floats, voxel-major
//   coords     : [L * 3] int32 at the input resolution (32^3 scaffold)
//   out_feats  : filled with [L_out * out_channels] floats, voxel-major
//   out_coords : filled with [L_out * 3] int32 at input_res * upscale()
TRELLIS2_API bool trellis2_shape_dec_decode( trellis2_shape_dec_model* m,
	const float*													   slat,
	int																   n_voxels,
	const int32_t*													   coords,
	std::vector<float>&												   out_feats,
	std::vector<int32_t>&											   out_coords,
	trellis2_shape_dec_taps*										   taps	 = nullptr,
	std::string*													   error = nullptr );

// Full shape decode plus the predicted sparse subdivision hierarchy, in decoder
// order (out_subs[0] is the first/coarsest upsample). This is the integrated
// image-to-3D texture path's guide_subs equivalent.
TRELLIS2_API bool trellis2_shape_dec_decode_with_subs( trellis2_shape_dec_model* m,
	const float*																 slat,
	int																			 n_voxels,
	const int32_t*																 coords,
	std::vector<float>&															 out_feats,
	std::vector<int32_t>&														 out_coords,
	std::vector<trellis2_subdiv_level>&											 out_subs,
	trellis2_shape_dec_taps*													 taps  = nullptr,
	std::string*																 error = nullptr );

// Predict just the subdivided coordinate set after `upsample_times` levels,
// without running the output layer — mirrors FlexiDualGridVaeDecoder.upsample(),
// used by the 1024 cascade to turn the LR (32^3) scaffold into candidate coords.
//   upsample_times : in [1, n_levels-1] (4 for the cascade: 32^3 -> 512^3)
//   out_coords     : filled with [L_out * 3] int32 at input_res * 2^upsample_times
TRELLIS2_API bool trellis2_shape_dec_upsample(
	trellis2_shape_dec_model* m, const float* slat, int n_voxels, const int32_t* coords, int upsample_times, std::vector<int32_t>& out_coords, std::string* error = nullptr );

/*****************************************************************************
** Public API – Shape-SLAT VAE encoder (FlexiDualGridVaeEncoder)
**
** The mirror of the shape decoder: a 6-channel dual grid ([offset(3),
** intersected(3)] per active voxel at resolution R) is downsampled 16x through
** SparseSpatial2Channel (S2C) blocks into a 32-channel latent at R/16 (the
** "shape SLAT"). This is used by the standalone arbitrary-mesh texturing API
** and its parity test. Integrated image-to-3D generation retains the original
** generated shape SLAT and shape-decoder subdivisions instead of performing a
** lossy decode/encode round trip.
*****************************************************************************/

struct trellis2_shape_enc_hparams {
	int32_t in_channels		= 6;	 // [offset(3), intersected(3)]
	int32_t latent_channels = 0;	 // 32
	int32_t n_levels		= 0;	 // 5
	int32_t channels[8]		= { 0 }; // [64, 128, 256, 512, 1024]
	int32_t num_blocks[8]	= { 0 }; // [0, 4, 8, 16, 4]
	float	norm_eps		= 1e-6f;
	int32_t file_type		= 0;
};

struct trellis2_shape_enc_model;

TRELLIS2_API trellis2_shape_enc_model*		   trellis2_shape_enc_load( const std::string& path, bool load_tensors = true, std::string* error = nullptr, const char* device = nullptr );

TRELLIS2_API void							   trellis2_shape_enc_free( trellis2_shape_enc_model* m );
TRELLIS2_API const char*					   trellis2_shape_enc_backend_name( const trellis2_shape_enc_model* m );
TRELLIS2_API const trellis2_shape_enc_hparams& trellis2_shape_enc_hparams_of( const trellis2_shape_enc_model* m );

// Encode the dual grid into the shape SLAT.
//   in6        : [N * 6] floats voxel-major = [offset(3), intersected(3)] (the
//                encoder subtracts 0.5 internally, matching the reference).
//   coords     : [N * 3] int32 at resolution R.
//   out_slat   : filled with [Nl * latent_channels] floats, voxel-major.
//   out_coords : filled with [Nl * 3] int32 at R/16.
//   out_subs   : filled with (n_levels-1) subdivision levels in decoder order.
TRELLIS2_API bool							   trellis2_shape_enc_encode( trellis2_shape_enc_model* m,
								 const float*														in6,
								 int																n_voxels,
								 const int32_t*														coords,
								 std::vector<float>&												out_slat,
								 std::vector<int32_t>&												out_coords,
								 std::vector<trellis2_subdiv_level>&								out_subs,
								 trellis2_shape_dec_taps*											taps  = nullptr,
								 std::string*														error = nullptr );

/*****************************************************************************
** Public API – Texture-SLAT VAE decoder (SparseUnetVaeDecoder, pred_subdiv=False)
**
** Architecturally the shape decoder with two differences: it does NOT predict
** the subdivision (it replays the encoder's recorded subdivision via `subs`)
** and it emits 6 PBR channels (base_color[3], metallic, roughness, alpha),
** already mapped to [0,1] (the reference's *0.5+0.5). Reuses the shape-decoder
** model struct/loader/driver.
*****************************************************************************/

// Load a trellis2-tex-dec GGUF (returns a shape_dec model with out_channels=6).
TRELLIS2_API trellis2_shape_dec_model*		   trellis2_tex_dec_load( const std::string& path, bool load_tensors = true, std::string* error = nullptr, const char* device = nullptr );

// Decode the (denormalized) texture SLAT into per-voxel PBR, replaying `subs`.
//   slat       : [L * latent_channels] tex SLAT, voxel-major.
//   coords     : [L * 3] latent voxels (== the shape SLAT coords).
//   subs       : shape decoder or encoder per-level subdivisions (decoder order).
//   out_feats  : [M * 6] PBR, voxel-major, already in [0,1].
//   out_coords : [M * 3] at R; a permutation-free reproduction of the encoder's
//                input voxel order (so PBR voxel v == mesh vertex v).
TRELLIS2_API bool							   trellis2_tex_dec_decode( trellis2_shape_dec_model* m,
								 const float*													  slat,
								 int															  n_voxels,
								 const int32_t*													  coords,
								 const std::vector<trellis2_subdiv_level>&						  subs,
								 std::vector<float>&											  out_feats,
								 std::vector<int32_t>&											  out_coords,
								 std::string*													  error = nullptr );

// Free VRAM (bytes) on the first GPU backend device, or 0 when there is no GPU
// (CPU-only build/host). Used to auto-place the shape decoder and to decide
// whether to free the flow DiTs before a decode. Cheap (a cudaMemGetInfo).
TRELLIS2_API size_t							   trellis2_gpu_free_vram();

// The numerics-affecting knobs (attention path, chunking, CUDA graphs) as this
// process resolved them, one key=value per line. For hosts that record what
// produced a result; see the comment at the definition for why the library has
// to be the one reporting it.
TRELLIS2_API std::string trellis2_effective_config();

// Attention path for the flow DiTs: 0 auto (size-gated), 1 exact, 2 flash.
// Process-wide, so it must be set before a generation and not during one; the
// "serialize calls on one pipeline" rule in the C API already requires that.
// Setting 0 restores whatever TRELLIS2_SDPA_EXACT / _FLASH say, so a host that
// does not know about the knob cannot override a deliberately pinned path.
TRELLIS2_API int		 trellis2_sdpa_mode();
TRELLIS2_API void		 trellis2_set_sdpa_mode( int mode );

/*****************************************************************************
** Public API – DINOv3 ViT-L/16 image-conditioning encoder
**
** The encoder that produces trellis2_dino_cond from an image, replacing the
** external dump_dinodata.py. Mirrors transformers' DINOv3ViTModel driven the
** way TRELLIS.2's DinoV3FeatureExtractor drives it: manual embeddings ->
** axial-2D-RoPE -> 24 layers, and an affine-free LayerNorm on the last layer
** output (the model's own final `norm` is NOT applied).
*****************************************************************************/

struct trellis2_dino_hparams {
	int32_t hidden_size			= 0;
	int32_t n_layers			= 0;
	int32_t n_heads				= 0;
	int32_t intermediate_size	= 0;
	int32_t patch_size			= 0;
	int32_t num_register_tokens = 0;
	float	layer_norm_eps		= 1e-5f;
	float	rope_theta			= 100.0f;
	float	image_mean[3]		= { 0.485f, 0.456f, 0.406f };
	float	image_std[3]		= { 0.229f, 0.224f, 0.225f };
	int32_t file_type			= 0; // 0=f32, 1=f16

	int32_t head_dim() const { return n_heads ? hidden_size / n_heads : 0; }
};

// Opaque handle to a loaded DINOv3 encoder (weights + metadata).
struct trellis2_dino_model;

// Optional per-layer activation taps for numerical validation. When a non-null
// pointer is passed to trellis2_dino_encode, every captured intermediate is
// appended as (name, row-major f32 buffer). Names match the reference dumper
// (scripts/dump_dino_reference.py): "embd", "l{i}.out", "cond", plus detail
// taps "l{i}.{norm1,attention,layer_scale1,norm2,mlp,layer_scale2}" for the
// first and last layer.
struct trellis2_dino_taps {
	std::vector<std::string>		names;
	std::vector<std::vector<float>> data;
};

// Load the DINOv3 encoder from a GGUF produced by convert_dino_to_gguf.py.
TRELLIS2_API trellis2_dino_model*		  trellis2_dino_load( const std::string& path, bool load_tensors = true, std::string* error = nullptr, const char* device = nullptr );

TRELLIS2_API void						  trellis2_dino_free( trellis2_dino_model* m );

TRELLIS2_API const char*				  trellis2_dino_backend_name( const trellis2_dino_model* m );

TRELLIS2_API const trellis2_dino_hparams& trellis2_dino_hparams_of( const trellis2_dino_model* m );

// Encode ImageNet-normalized pixel values into the conditioning tensor.
//
//   pixel_values : [3 * S * S] floats, CHW ((x - mean) / std already applied) —
//                  exactly the flattened [1, 3, S, S] the Python forward sees.
//   image_size   : S; must be a multiple of patch_size (512 for the TRELLIS.2
//                  "512" pipeline -> 1 CLS + 4 register + 1024 patch tokens).
//   out          : the conditioning tensor, shape {1, N, C}.
//   taps         : optional activation capture for validation (see above).
//
// Returns false on failure and (if `error`) fills it with a reason.
TRELLIS2_API bool trellis2_dino_encode( trellis2_dino_model* m, const float* pixel_values, int image_size, trellis2_dino_cond& out, trellis2_dino_taps* taps = nullptr, std::string* error = nullptr );

// Convenience wrapper: 8-bit RGB (HWC, [S*S*3]) -> /255 -> ImageNet normalize
// -> trellis2_dino_encode.
TRELLIS2_API bool trellis2_dino_encode_rgb( trellis2_dino_model* m, const uint8_t* rgb, int image_size, trellis2_dino_cond& out, std::string* error = nullptr );

/*****************************************************************************
** Public API – Image preprocessing (TRELLIS.2 pipeline.preprocess_image)
**
** For an RGBA input with a meaningful alpha channel (the "has_alpha" path):
** downscale so max(W,H) <= 1024,
** square-crop around the alpha>0.8 bounding box, premultiply onto black, and
** LANCZOS-resize to out_size x out_size RGB. The resampler reproduces PIL's
** fixed-point uint8 Lanczos-3 so results match the Python pipeline.
*****************************************************************************/

enum trellis2_background_mode {
	TRELLIS2_BACKGROUND_AUTO  = 0,
	TRELLIS2_BACKGROUND_KEEP  = 1,
	TRELLIS2_BACKGROUND_BLACK = 2,
	TRELLIS2_BACKGROUND_WHITE = 3,
};

// Replace a near-black/near-white background connected to the image border
// with softly feathered alpha. AUTO leaves images with meaningful existing
// alpha or a non-black/non-white border unchanged. Returns pixels whose alpha
// changed, or -1 for invalid arguments.
TRELLIS2_API int					   trellis2_remove_solid_background_rgba( uint8_t* rgba, int w, int h, int mode = TRELLIS2_BACKGROUND_AUTO );

// rgba: [h * w * 4] bytes. On success out_rgb has out_size*out_size*3 bytes.
// Fails if the image has no pixels with alpha > 0.8*255.
TRELLIS2_API bool					   trellis2_preprocess_rgba( const uint8_t* rgba, int w, int h, int out_size, std::vector<uint8_t>& out_rgb, std::string* error = nullptr );

/*****************************************************************************
** Public API – DINOv3 conditioning (.dinodata)
*****************************************************************************/

// Load a .dinodata file produced by dump_dinodata.py.
// Returns true on success; on failure returns false and (if `error` != nullptr)
// fills it with a human-readable reason. On success `out` is fully populated.
TRELLIS2_API bool					   trellis2_load_dinodata( const std::string& path, trellis2_dino_cond& out, std::string* error = nullptr );

// Compute fingerprints over a loaded cond (min/max/mean/sum/l2/count).
TRELLIS2_API trellis2_dino_fingerprint trellis2_dino_fingerprints( const trellis2_dino_cond& cond );

// Write a cond to a .dinodata file (the format trellis2_load_dinodata reads).
TRELLIS2_API bool					   trellis2_save_dinodata( const std::string& path, const trellis2_dino_cond& cond, std::string* error = nullptr );
