#pragma once
//
// mesh_export — CUDA-free port of o_voxel.postprocess.to_glb: take the demo's
// dense per-vertex-PBR mesh and produce a portable glTF 2.0 binary (GLB). The
// reference does this on the GPU (CuMesh simplify/unwrap/BVH, nvdiffrast
// raster, flex_gemm grid_sample); here every stage is pure C++:
//
//   preserve topology -> optional component cleanup -> standard COLOR_0 vertex
//   material + retained custom metallic/roughness attribute -> GLB.
//
// A practical fixed-size atlas cannot give millions of preserved triangles
// enough texels, and the dual-grid geometry is heavily non-manifold. glTF vertex
// colour is therefore both more faithful and substantially smaller. Set
// T2GLB_XATLAS to opt into a conventional image atlas on clean meshes. No ggml /
// CUDA dependency: plain float/int arrays.

#include "trellis2.h" // TRELLIS2_API

#include <cstdint>
#include <string>
#include <vector>

namespace t2glb
{

enum class ComponentFilter {
	RemoveTiny	= 0, // preserve meaningful disconnected parts
	KeepLargest = 1, // retain only the component with the most triangles
	KeepAll		= 2	 // input is already prepared; do not filter again
};

struct MeshExportOptions {
	int				texture_size = 2048; // opt-in T2GLB_XATLAS width/height
	int				padding		 = 2;	 // xatlas chart padding (texels; T2GLB_XATLAS)
	int				dilate		 = 6;	 // gutter dilation passes (kills UV seams)
	ComponentFilter components	 = ComponentFilter::RemoveTiny;
};

// Geometry/material streams after the same component filtering used by
// mesh_to_glb. Valid source triangles retain their original polygon density.
struct PreparedMesh {
	std::vector<float>	 verts;
	std::vector<float>	 normals;
	std::vector<int32_t> tris;
	std::vector<float>	 pbr;
};

TRELLIS2_API bool prepare_mesh( const float* verts, int nv, const int32_t* tris, int nt, const float* pbr, const MeshExportOptions& opt, PreparedMesh& out, std::string& err );

// Optional CPU print-remesh path backed by CGAL Alpha Wrap 3. The ratios are
// fractions of the component-filtered mesh's bounding-box diagonal. The result
// is guaranteed by Alpha Wrap to be closed, oriented, intersection-free and
// 2-manifold. Wrapping creates a new enclosing surface, so a textured source is
// sampled onto the wrap vertices (approximate per-vertex preview); the sharper
// per-texel rebake stays in mesh_to_projected_glb for the GLB download.
TRELLIS2_API bool print_remesh_available();
TRELLIS2_API bool prepare_print_mesh(
	const float* verts, int nv, const int32_t* tris, int nt, const float* pbr, const MeshExportOptions& opt, float alpha_ratio, float offset_ratio, PreparedMesh& out, std::string& err );

// Export a dense per-vertex-PBR mesh as a standard vertex-coloured GLB.
//
//   verts   3*nv  vertex positions (mesh/world space, as fdg::extract emits)
//   tris    3*nt  triangle vertex indices
//   pbr     6*nv  base_color rgb, metallic, roughness, alpha
//                 (null -> untextured grey)
//
// On success fills `out` with the GLB bytes and returns true. On failure returns
// false with a message in `err`. Not reentrant (Simplify.h uses global state):
// serialized internally by a mutex.
TRELLIS2_API bool mesh_to_glb( const float* verts, int nv, const int32_t* tris, int nt, const float* pbr, const MeshExportOptions& opt, std::vector<uint8_t>& out, std::string& err );

// UV-unwrap `target` and bake its atlas by projecting each covered texel onto
// the closest triangle of the dense PBR `source`.  Intended for assigning the
// generated material to Alpha Wrap geometry; always uses xatlas and the CGAL
// CPU closest-surface backend regardless of T2GLB_XATLAS.
TRELLIS2_API bool mesh_to_projected_glb( const float* target_verts,
	int												  target_nv,
	const int32_t*									  target_tris,
	int												  target_nt,
	const float*									  source_verts,
	int												  source_nv,
	const int32_t*									  source_tris,
	int												  source_nt,
	const float*									  source_pbr,
	const MeshExportOptions&						  opt,
	std::vector<uint8_t>&							  out,
	std::string&									  err );

} // namespace t2glb
