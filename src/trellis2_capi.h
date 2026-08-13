/*
** trellis2_capi.h — flat C ABI for embedding trellis2 in non-C++ hosts
** (the Go demo server dlopens libtrellis2.so and binds these by name).
**
** ABI version bumps whenever any signature or struct layout here changes.
*/
#pragma once

#include <stdint.h>

#ifdef TRELLIS2_SHARED
	#if defined( _WIN32 ) && !defined( __MINGW32__ )
		#ifdef TRELLIS2_BUILD
			#define TRELLIS2_CAPI __declspec( dllexport )
		#else
			#define TRELLIS2_CAPI __declspec( dllimport )
		#endif
	#else
		#define TRELLIS2_CAPI __attribute__( ( visibility( "default" ) ) )
	#endif
#else
	#define TRELLIS2_CAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 16: T2_PIPE_1536 / T2_CAP_1536; t2_mesh_grid_resolution added. */
#define T2_CAPI_ABI_VERSION 16

TRELLIS2_CAPI int t2_abi_version();

/* Pipeline stages reported by the progress callback. */
enum t2_stage {
	T2_STAGE_PREPROCESS	  = 0, /* image decode + crop/premultiply/resize */
	T2_STAGE_DINO		  = 1, /* conditioning encoder                   */
	T2_STAGE_SS_FLOW	  = 2, /* sparse-structure flow sampling (steps) */
	T2_STAGE_SS_DEC		  = 3, /* occupancy decoder -> voxel scaffold    */
	T2_STAGE_SLAT_FLOW	  = 4, /* shape-SLAT flow sampling (steps)       */
	T2_STAGE_SHAPE_DEC	  = 5, /* shape decoder -> dual-grid fields      */
	T2_STAGE_MESH		  = 6, /* mesh extraction                        */
	T2_STAGE_UPSAMPLE	  = 7, /* cascade: LR slat -> HR voxel scaffold  */
	T2_STAGE_SLAT_FLOW_HR = 8, /* cascade: 1024-model shape-SLAT flow     */
	T2_STAGE_SHAPE_DEC_HR = 9, /* cascade: HR shape decoder (1024/1536^3) */
	T2_STAGE_TEXTURE	  = 10 /* PBR texture: flow + guided decode      */
};

/* Pipeline type for t2_generate. */
enum t2_pipeline_type {
	T2_PIPE_AUTO   = 0, /* cascade if available, else 512 fine, else coarse */
	T2_PIPE_COARSE = 1, /* 64^3 occupancy -> marching cubes preview        */
	T2_PIPE_512	   = 2, /* 512 fine dual-grid                              */
	T2_PIPE_1024   = 3, /* 1024 cascade                                    */
	/* 1536 cascade: same checkpoints as T2_PIPE_1024 on a 96^3 HR scaffold.
	** The HR token budget can step the achieved resolution back down toward
	** 1024 in units of 128 (see T2_STAGE_UPSAMPLE below and
	** t2_mesh_grid_resolution). Needs a card that can hold the larger decode;
	** T2_PIPE_AUTO deliberately never selects it. */
	T2_PIPE_1536   = 4
};

/* Solid-background handling before the alpha-bbox crop. */
enum t2_background_mode {
	T2_BACKGROUND_AUTO	= 0, /* detect border-connected near-black/near-white */
	T2_BACKGROUND_KEEP	= 1, /* preserve the decoded alpha exactly             */
	T2_BACKGROUND_BLACK = 2, /* force removal of border-connected near-black   */
	T2_BACKGROUND_WHITE = 3	 /* force removal of border-connected near-white   */
};

/* Capability bits reported by t2_pipeline_caps (loaded qualities/features).
** T2_CAP_1536 accompanies T2_CAP_1024 — the tier reuses the same checkpoints,
** so having the models is what makes it available. Whether the card can hold
** the 1536^3 decode is not part of the capability; that shows up as a decode
** failure, not as a silently downgraded run. */
enum t2_caps { T2_CAP_COARSE = 1, T2_CAP_512 = 2, T2_CAP_1024 = 4, T2_CAP_TEXTURE = 8, T2_CAP_1536 = 16 };

/* Load-time flags for t2_pipeline_load. */
enum t2_load_flags {
	T2_LOAD_LOW_VRAM = 1 /* reserved: load DiTs on demand (follow-on)     */
};

/* step/total are meaningful for T2_STAGE_SS_FLOW; other stages send 0/0 at
** entry. Called from the generating thread.
**
** One exception, so a reduced cascade run is visible rather than silent:
** T2_STAGE_UPSAMPLE fires a SECOND time once the HR token budget has settled,
** with step = the achieved resolution and total = the requested one. They are
** equal whenever nothing was reduced, and both are >= 1024, so a host that
** only understands step counters still shows a sane "0/0 then 1024/1536". */
typedef void ( *t2_progress_fn )( void* user, int stage, int step, int total );

/* Optional live intermediate-preview callback. During generation the host
** receives self-describing geometry blobs (currently "T2VOX01": a voxel set —
** magic[8], u32 res, u32 nvox, u16[3*nvox] coords in [0,res)) for the current
** stage, so a viewer can show the shape emerging in 3D. `data` is valid only
** during the call (copy it). Fires on the generating thread. NULL disables it.
** See t2_generate. */
typedef void ( *t2_preview_fn )( void* user, int stage, int step, int total, const void* data, int len );

typedef struct t2_pipeline	  t2_pipeline;
typedef struct t2_mesh_result t2_mesh_result;

/* Load the pipeline models. Optional models (pass NULL/"" to omit) select the
** available qualities:
**   - slat_flow_gguf + shape_dec_gguf present  -> 512 fine dual-grid
**   - + slat_hr_flow_gguf present               -> 1024 cascade (reuses shape_dec)
**   - shape_enc_gguf + tex_dec_gguf + tex_flow_gguf -> PBR texturing
**   - neither pair                              -> coarse marching-cubes preview
** `flags` is a bitmask of t2_load_flags (0 for standard resident loading).
** On failure returns NULL and, if err != NULL, writes a reason into err. */
TRELLIS2_CAPI t2_pipeline*	  t2_pipeline_load( const char* dino_gguf,
	   const char*											ss_flow_gguf,
	   const char*											ss_dec_gguf,
	   const char*											slat_flow_gguf,
	   const char*											slat_hr_flow_gguf,
	   const char*											shape_dec_gguf,
	   /* PBR texturing (optional; NULL/"" to disable). The tex
	   ** models are loaded lazily per-generate, not held resident.
	   ** The validated generation path uses shape_enc_gguf to
	   ** re-encode the decoded dual grid before texture flow. */
	   const char*											shape_enc_gguf,
	   const char*											tex_dec_gguf,
	   const char*											tex_flow_gguf,
	   const char*											tex_flow_hr_gguf,
	   int													flags,
	   char*												err,
	   int													err_len );

/* Bitmask of t2_caps: which mesh qualities/features this pipeline can produce. */
TRELLIS2_CAPI int			  t2_pipeline_caps( t2_pipeline* p );
/* Back-compat: 1 if any fine (512/1024) path is available, else 0. */
TRELLIS2_CAPI int			  t2_pipeline_is_fine( t2_pipeline* p );
TRELLIS2_CAPI void			  t2_pipeline_free( t2_pipeline* p );
TRELLIS2_CAPI const char*	  t2_pipeline_backend( t2_pipeline* p );

/* image bytes (PNG/JPEG/...; anything stb_image decodes) -> triangle mesh.
** pipeline_type is a t2_pipeline_type (T2_PIPE_AUTO picks the best available).
** background_mode is a t2_background_mode (normally T2_BACKGROUND_AUTO).
** steps <= 0, guidance < 0, and texture_steps <= 0 select the pipeline defaults
** (12 / 7.5 / 12 respectively).
** `preview` (may be NULL) streams live intermediate 3D previews as the sparse
** structure emerges; `preview_user` is passed back to it. The T2_PREVIEW_STRIDE
** env var (default: ~4 previews across the SS steps) tunes the per-step cadence.
** NOT thread-safe per pipeline: serialize calls on one t2_pipeline. */
TRELLIS2_CAPI t2_mesh_result* t2_generate( t2_pipeline* p,
	const void*											image_bytes,
	int													image_len,
	int													pipeline_type,
	int													background_mode,
	uint64_t											seed,
	int													steps,
	float												guidance,
	int													texture_steps,
	t2_progress_fn										progress,
	void*												user,
	t2_preview_fn										preview,
	void*												preview_user,
	char*												err,
	int													err_len );

/* Mesh accessors. Vertices are in a centered unit cube ([-0.5, 0.5]^3, same
** axes as the voxel grid); normals are per-vertex unit vectors. Buffers stay
** valid until t2_mesh_free. */
TRELLIS2_CAPI int			  t2_mesh_n_verts( const t2_mesh_result* r );
TRELLIS2_CAPI int			  t2_mesh_n_tris( const t2_mesh_result* r );
TRELLIS2_CAPI const float*	  t2_mesh_verts( const t2_mesh_result* r );	  /* 3*n_verts   */
TRELLIS2_CAPI const float*	  t2_mesh_normals( const t2_mesh_result* r ); /* 3*n_verts   */
TRELLIS2_CAPI const int*	  t2_mesh_tris( const t2_mesh_result* r );	  /* 3*n_tris    */
/* Per-vertex PBR (6*n_verts: base_color rgb, metallic, roughness, alpha), or NULL when
** the mesh is untextured. t2_mesh_has_pbr reports availability. */
TRELLIS2_CAPI int			  t2_mesh_has_pbr( const t2_mesh_result* r );
TRELLIS2_CAPI const float*	  t2_mesh_pbr( const t2_mesh_result* r ); /* 6*n_verts   */
/* Grid resolution this mesh was actually extracted at: 64 (coarse), 512, 1024,
** 1536, or an intermediate multiple of 128 when the cascade token budget
** stepped a 1536 request down. 0 for meshes that did not come from t2_generate
** (t2_prepare_mesh and the remesh entry points), which only transform geometry
** and have no tier of their own. */
TRELLIS2_CAPI int			  t2_mesh_grid_resolution( const t2_mesh_result* r );
TRELLIS2_CAPI void			  t2_mesh_free( t2_mesh_result* r );

/* Prepare the exact component-cleaned geometry used for export so hosts can preview it.
** component_filter: 0 removes only tiny islands; 1 keeps the largest connected
** component; 2 keeps all components. The returned mesh owns copied PBR and is
** freed with t2_mesh_free. */
TRELLIS2_CAPI t2_mesh_result* t2_prepare_mesh( const float* verts, int n_verts, const int* tris, int n_tris, const float* pbr, int component_filter, char* err, int err_len );

/* Optional CGAL Alpha Wrap print remeshing. Availability is fixed at build
** time. alpha_ratio and offset_ratio are fractions of the component-filtered
** input bounding-box diagonal (recommended starting point: 0.01 and
** 0.01/30). The returned geometry is watertight, oriented, intersection-free,
** 2-manifold, and geometry-only (no PBR) because wrapping creates new vertices.
** Use t2_bake_projected_glb to transfer source PBR onto its UV atlas. */
TRELLIS2_CAPI int			  t2_print_remesh_available();
TRELLIS2_CAPI t2_mesh_result* t2_prepare_print_mesh(
	const float* verts, int n_verts, const int* tris, int n_tris, const float* pbr, int component_filter, float alpha_ratio, float offset_ratio, char* err, int err_len );

/* Optional AutoRemesher quad remeshing. Availability is fixed at build time and
** depends on Eigen 5.x, not on CGAL. target_quads is a density hint rather than
** a face count (it feeds a voxel size, and adaptivity moves the result a long
** way from it); adaptivity is 0..1, <0 keeps the default. The returned geometry
** is triangulated from a quad-dominant remesh and carries PBR sampled onto its
** new vertices.
**
** NOT watertight: AutoRemesher can drop islands whose parameterization fails
** and its quad extraction can leave boundaries. Use t2_prepare_print_mesh when
** the result has to be printable. The call fails rather than returning a
** half-remeshed model when too much surface area was lost. */
/* "cgal" or "tinybvh": which closest-surface backend project_pbr and the
** projected GLB bake use. Fixed at build time. Reporting only - both produce a
** result, they differ in which library does the nearest-surface search. The
** returned pointer is a static string and is never freed. */
TRELLIS2_CAPI const char*	  t2_projection_backend( void );

/* Sub-stage durations of the most recent t2_bake_glb / t2_bake_projected_glb
** call, in seconds. On dense meshes the bake is the largest single export cost
** and the UV unwrap usually dominates it, so hosts should report the split
** rather than one opaque number. Any pointer may be NULL. Values are from the
** last bake in this process; the bake is internally serialized. */
TRELLIS2_CAPI void			  t2_last_bake_timings( float* out_unwrap, float* out_rasterize, float* out_projection, float* out_texel_fill, float* out_encode );

TRELLIS2_CAPI int			  t2_quad_remesh_available( void );
TRELLIS2_CAPI t2_mesh_result* t2_prepare_quad_mesh(
	const float* verts, int n_verts, const int* tris, int n_tris, const float* pbr, int component_filter, int target_quads, float adaptivity, char* err, int err_len );
/* Quality of the last t2_prepare_quad_mesh result, for hosts that want to
** report it. Written only on success; any pointer may be NULL. */
TRELLIS2_CAPI void	   t2_quad_mesh_stats( const t2_mesh_result* r, int* out_quads, int* out_triangles, int* out_ngons, int* out_boundary_edges, float* out_area_retained );

/* Bake a mesh into a portable UV-atlas-textured GLB (glTF 2.0 binary): optional
** component cleanup -> UV unwrap -> per-texel PBR bake (from the dense
** per-vertex attributes) -> gutter inpaint -> glTF. All CPU, no CUDA.
**   verts  3*n_verts, tris  3*n_tris, pbr  6*n_verts (base_color rgb, metallic,
**   roughness, alpha) or NULL for an untextured grey bake.
**   texture_size  square atlas resolution hint (e.g. 2048; <=0 -> default).
**   component_filter  0 remove tiny islands, 1 largest only, 2 keep all.
** Operates on raw arrays so hosts can bake straight from their own buffers.
** On success returns a malloc'd GLB buffer (free with t2_free_buffer) and writes
** its length to *out_len; on failure returns NULL with a reason in err. */
TRELLIS2_CAPI uint8_t* t2_bake_glb( const float* verts, int n_verts, const int* tris, int n_tris, const float* pbr, int texture_size, int component_filter, int* out_len, char* err, int err_len );

/* Bake a UV-atlas PBR GLB for replacement geometry by closest-surface
** projection from a dense source mesh. Every covered target atlas texel is
** projected to a source triangle and receives barycentrically interpolated
** source PBR. This is the CPU counterpart of upstream's cuBVH rebake, backed by
** tinybvh, so it is available in every build — unlike t2_prepare_print_mesh,
** which still needs CGAL.
** source_component_filter has the same 0/1/2 values as component_filter above.
**
** normal_map (nonzero = on, and the recommended default) additionally bakes a
** tangent-space normal map from the source's shading normals and exports the
** MikkTSpace-compatible TANGENT attribute it was baked against. That is what
** carries the dense mesh's surface detail onto replacement geometry, which
** would otherwise have to be normal-baked by hand in a DCC tool. It costs a
** third full-resolution PNG in the GLB. Use t2_last_normal_map_stats to report
** how much of it the closest-surface search had to leave flat. */
TRELLIS2_CAPI uint8_t* t2_bake_projected_glb( const float* target_verts,
	int													   target_n_verts,
	const int*											   target_tris,
	int													   target_n_tris,
	const float*										   source_verts,
	int													   source_n_verts,
	const int*											   source_tris,
	int													   source_n_tris,
	const float*										   source_pbr,
	int													   texture_size,
	int													   source_component_filter,
	int													   normal_map,
	int*												   out_len,
	char*												   err,
	int													   err_len );

/* Normal map quality of the most recent t2_bake_projected_glb call. A
** closest-point search is not a bake cage: on thin geometry the nearest surface
** is the far side of a wall, which would invert the shading lobe. Those texels
** are left flat and counted here, so a host can surface the loss instead of
** shipping a quietly wrong map. Both outputs are 0 when no normal map was
** baked; either pointer may be NULL. */
TRELLIS2_CAPI void	   t2_last_normal_map_stats( int* out_covered_texels, int* out_rejected_texels );
TRELLIS2_CAPI void	   t2_free_buffer( uint8_t* buf );

/* Image decode + TRELLIS.2 preprocessing only (no models). out_rgb must hold
** out_size*out_size*3 bytes. Returns 0 on success, nonzero on failure (reason
** in err). This is the untrusted-input surface targeted by the fuzzers. */
TRELLIS2_CAPI int	   t2_preprocess_image_bytes( const void* image_bytes, int image_len, int out_size, unsigned char* out_rgb, char* err, int err_len );

#ifdef __cplusplus
}
#endif
