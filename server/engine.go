package main

// engine.go — in-process FFI to libtrellis2 (libtrellis2.so on Linux/macOS,
// libtrellis2.dll on Windows) via purego (no cgo), following the depth-anything.cpp
// server pattern. The C ABI is trellis2_capi.h; the t2_abi_version binding
// guards against header/library drift.
//
// The platform-specific library handle is acquired in engine_lib_{unix,windows}.go
// (build-tagged) — purego.Dlopen on Unix, syscall.LoadLibrary on Windows.

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const abiVersion = 19

// glbOptions mirrors struct t2_glb_options field for field (five 4-byte fields,
// no padding on either side). Passed by pointer, which is what keeps both bake
// bindings under purego's 15-argument ceiling — the flat lists had reached it.
type glbOptions struct {
	TextureSize     int32
	ComponentFilter int32
	NormalMap       int32
	ExportFlags     int32
	UnitScale       float32
}

// Attention path for the flow DiTs (enum t2_attention_mode). Exact accumulates
// in F32; flash is faster but ggml's HIP kernels accumulate in F16, which costs
// thin structure at the cascade tiers.
const (
	attentionAuto  = 0
	attentionExact = 1
	attentionFlash = 2
)

// Mirrors struct t2_gen_options. Seed leads so the 8-byte field is aligned
// without interior padding on either side of the boundary; keep the order.
type genOptions struct {
	Seed           uint64
	PipelineType   int32
	BackgroundMode int32
	Steps          int32
	TextureSteps   int32
	AttentionMode  int32
	Guidance       float32
}

// Progress stages (enum t2_stage).
const (
	stagePreprocess = 0
	stageDino       = 1
	stageSSFlow     = 2
	stageSSDec      = 3
	stageSLATFlow   = 4
	stageShapeDec   = 5
	stageMesh       = 6
	stageUpsample   = 7
	stageSLATFlowHR = 8
	stageShapeDecHR = 9
	stageTexture    = 10
)

var stageNames = map[int]string{
	stagePreprocess: "preprocess",
	stageDino:       "encoding image (DINOv3)",
	stageSSFlow:     "sampling sparse structure",
	stageSSDec:      "decoding occupancy",
	stageSLATFlow:   "sampling shape SLAT",
	stageShapeDec:   "decoding shape",
	stageMesh:       "extracting mesh",
	stageUpsample:   "upsampling scaffold",
	// The HR stages keep their "(1024)" label at every cascade tier: it names
	// the model doing the work, which is the 1024 checkpoint at 1536 too. The
	// grid the run actually landed on is reported separately, as job.resolution.
	stageSLATFlowHR: "sampling shape SLAT (1024)",
	stageShapeDecHR: "decoding shape (1024)",
	stageTexture:    "sampling material",
}

// Pipeline types (enum t2_pipeline_type) and capability bits (enum t2_caps).
const (
	pipeAuto   = 0
	pipeCoarse = 1
	pipe512    = 2
	pipe1024   = 3
	pipe1536   = 4

	capCoarse  = 1
	cap512     = 2
	cap1024    = 4
	capTexture = 8
	cap1536    = 16

	backgroundAuto  = 0
	backgroundKeep  = 1
	backgroundBlack = 2
	backgroundWhite = 3
)

type engine struct {
	// inference is not thread-safe: one generation at a time.
	mu      sync.Mutex
	stateMu sync.RWMutex

	pipeline uintptr
	backend  string
	caps     int  // bitmask of t2_caps
	textured bool // PBR texturing enabled
	models   engineModels

	abiVersion      func() int32
	pipelineLoad    func(dino, flow, dec, slat, slatHR, shapeDec, shapeEnc, texDec, texFlow, texFlowHR string, flags int32, err unsafe.Pointer, errLen int32) uintptr
	pipelineFree    func(p uintptr)
	pipelineBackend func(p uintptr) string
	pipelineCaps    func(p uintptr) int32
	runConfig       func(p uintptr) string
	generate        func(p uintptr, img unsafe.Pointer, imgLen int32, opts *genOptions,
		cb uintptr, user unsafe.Pointer, preview uintptr, previewUser unsafe.Pointer,
		err unsafe.Pointer, errLen int32) uintptr
	meshNVerts   func(r uintptr) int32
	meshNTris    func(r uintptr) int32
	meshVerts    func(r uintptr) uintptr
	meshNormals  func(r uintptr) uintptr
	meshTris     func(r uintptr) uintptr
	meshHasPBR   func(r uintptr) int32
	meshPBR      func(r uintptr) uintptr
	meshGridRes  func(r uintptr) int32
	meshFree     func(r uintptr)
	prepareMeshC func(verts unsafe.Pointer, nv int32, tris unsafe.Pointer, nt int32, pbr unsafe.Pointer,
		componentFilter int32, err unsafe.Pointer, errLen int32) uintptr
	projectionBackendC  func() uintptr
	lastBakeTimings     func(unwrap, rasterize, projection, texelFill, encode unsafe.Pointer)
	quadRemeshAvailable func() int32
	prepareQuadMeshC    func(verts unsafe.Pointer, nv int32, tris unsafe.Pointer, nt int32, pbr unsafe.Pointer,
		componentFilter, targetQuads int32, adaptivity float32,
		err unsafe.Pointer, errLen int32) uintptr
	quadMeshStats func(r uintptr, quads, triangles, ngons, boundaryEdges unsafe.Pointer, areaRetained unsafe.Pointer)

	printRemeshAvailable func() int32
	preparePrintMeshC    func(verts unsafe.Pointer, nv int32, tris unsafe.Pointer, nt int32, pbr unsafe.Pointer,
		componentFilter int32, alphaRatio, offsetRatio float32,
		err unsafe.Pointer, errLen int32) uintptr
	bakeGLB func(verts unsafe.Pointer, nv int32, tris unsafe.Pointer, nt int32, pbr unsafe.Pointer,
		opts unsafe.Pointer, outLen unsafe.Pointer, err unsafe.Pointer, errLen int32) uintptr
	bakeProjectedGLB func(targetVerts unsafe.Pointer, targetNV int32, targetTris unsafe.Pointer, targetNT int32,
		sourceVerts unsafe.Pointer, sourceNV int32, sourceTris unsafe.Pointer, sourceNT int32,
		sourcePBR unsafe.Pointer, opts unsafe.Pointer,
		outLen unsafe.Pointer, err unsafe.Pointer, errLen int32) uintptr
	lastNormalMapStats func(coveredTexels, rejectedTexels unsafe.Pointer)
	freeBuffer         func(buf uintptr)
}

// engineModels retains only the paths needed to recreate a freed pipeline. The
// GGUF contents themselves remain owned by the C pipeline while it is loaded.
type engineModels struct {
	dino, flow, dec                      string
	slat, slatHR, shapeDec               string
	shapeEnc, texDec, texFlow, texFlowHR string
}

// progressSink receives per-stage/step updates for the currently running
// generation. Exactly one generation runs at a time (engine.mu), so a single
// global callback + current sink is safe.
var (
	progressMu   sync.Mutex
	progressSink func(stage, step, total int)
	previewSink  func(stage, step, total int, blob []byte)
)

var (
	progressCallback uintptr // created once; purego callbacks are permanent
	previewCallback  uintptr
)

// slatGGUF/shapeDecGGUF may be "" for the coarse path; slatHRGGUF may be "" to
// disable the 1024 cascade (512 fine only).
func newEngine(libPath, dinoGGUF, flowGGUF, decGGUF, slatGGUF, slatHRGGUF, shapeDecGGUF,
	shapeEncGGUF, texDecGGUF, texFlowGGUF, texFlowHRGGUF string, startUnloaded bool) (*engine, error) {
	lib, err := openLib(libPath)
	if err != nil {
		return nil, fmt.Errorf("load library %s: %w", libPath, err)
	}

	e := &engine{models: engineModels{
		dino: dinoGGUF, flow: flowGGUF, dec: decGGUF,
		slat: slatGGUF, slatHR: slatHRGGUF, shapeDec: shapeDecGGUF,
		shapeEnc: shapeEncGGUF, texDec: texDecGGUF,
		texFlow: texFlowGGUF, texFlowHR: texFlowHRGGUF,
	}}
	purego.RegisterLibFunc(&e.abiVersion, lib, "t2_abi_version")
	if got := e.abiVersion(); got != abiVersion {
		return nil, fmt.Errorf("ABI mismatch: library reports %d, server built for %d", got, abiVersion)
	}
	purego.RegisterLibFunc(&e.pipelineLoad, lib, "t2_pipeline_load")
	purego.RegisterLibFunc(&e.pipelineFree, lib, "t2_pipeline_free")
	purego.RegisterLibFunc(&e.pipelineBackend, lib, "t2_pipeline_backend")
	purego.RegisterLibFunc(&e.pipelineCaps, lib, "t2_pipeline_caps")
	purego.RegisterLibFunc(&e.runConfig, lib, "t2_run_config")
	purego.RegisterLibFunc(&e.generate, lib, "t2_generate")
	purego.RegisterLibFunc(&e.meshNVerts, lib, "t2_mesh_n_verts")
	purego.RegisterLibFunc(&e.meshNTris, lib, "t2_mesh_n_tris")
	purego.RegisterLibFunc(&e.meshVerts, lib, "t2_mesh_verts")
	purego.RegisterLibFunc(&e.meshNormals, lib, "t2_mesh_normals")
	purego.RegisterLibFunc(&e.meshTris, lib, "t2_mesh_tris")
	purego.RegisterLibFunc(&e.meshHasPBR, lib, "t2_mesh_has_pbr")
	purego.RegisterLibFunc(&e.meshPBR, lib, "t2_mesh_pbr")
	purego.RegisterLibFunc(&e.meshGridRes, lib, "t2_mesh_grid_resolution")
	purego.RegisterLibFunc(&e.meshFree, lib, "t2_mesh_free")
	purego.RegisterLibFunc(&e.prepareMeshC, lib, "t2_prepare_mesh")
	purego.RegisterLibFunc(&e.projectionBackendC, lib, "t2_projection_backend")
	purego.RegisterLibFunc(&e.lastBakeTimings, lib, "t2_last_bake_timings")
	purego.RegisterLibFunc(&e.quadRemeshAvailable, lib, "t2_quad_remesh_available")
	purego.RegisterLibFunc(&e.prepareQuadMeshC, lib, "t2_prepare_quad_mesh")
	purego.RegisterLibFunc(&e.quadMeshStats, lib, "t2_quad_mesh_stats")
	purego.RegisterLibFunc(&e.printRemeshAvailable, lib, "t2_print_remesh_available")
	purego.RegisterLibFunc(&e.preparePrintMeshC, lib, "t2_prepare_print_mesh")
	purego.RegisterLibFunc(&e.bakeGLB, lib, "t2_bake_glb")
	purego.RegisterLibFunc(&e.bakeProjectedGLB, lib, "t2_bake_projected_glb")
	purego.RegisterLibFunc(&e.lastNormalMapStats, lib, "t2_last_normal_map_stats")
	purego.RegisterLibFunc(&e.freeBuffer, lib, "t2_free_buffer")

	if progressCallback == 0 {
		progressCallback = purego.NewCallback(func(user unsafe.Pointer, stage, step, total int32) uintptr {
			progressMu.Lock()
			sink := progressSink
			progressMu.Unlock()
			if sink != nil {
				sink(int(stage), int(step), int(total))
			}
			return 0
		})
	}
	if previewCallback == 0 {
		// Live intermediate-preview blobs (T2VOX01 voxel sets). `data` is only
		// valid during the call, so copy before handing it to the sink.
		previewCallback = purego.NewCallback(func(user unsafe.Pointer, stage, step, total int32,
			data unsafe.Pointer, length int32) uintptr {
			progressMu.Lock()
			sink := previewSink
			progressMu.Unlock()
			if sink != nil && data != nil && length > 0 {
				blob := make([]byte, int(length))
				copy(blob, unsafe.Slice((*byte)(data), int(length)))
				sink(int(stage), int(step), int(total), blob)
			}
			return 0
		})
	}

	if startUnloaded {
		e.backend = "GPU (models unloaded)"
		if os.Getenv("TRELLIS2_DEVICE") == "cpu" {
			e.backend = "CPU (models unloaded)"
		}
		e.caps = configuredCaps(e.models)
		e.textured = e.caps&capTexture != 0
	} else {
		if err := e.loadLocked(); err != nil {
			return nil, err
		}
	}
	return e, nil
}

// configuredCaps mirrors t2_pipeline_caps from the model paths that main has
// already existence-checked. It keeps /api/info and the quality picker useful
// before a lazy first pipeline load.
func configuredCaps(m engineModels) int {
	caps := capCoarse
	if m.slat != "" && m.shapeDec != "" {
		caps |= cap512
		if m.slatHR != "" {
			caps |= cap1024 | cap1536
		}
		if m.shapeEnc != "" && m.texDec != "" && m.texFlow != "" {
			caps |= capTexture
		}
	}
	return caps
}

// loadLocked recreates the pipeline after an idle unload. e.mu must be held by
// callers after construction.
func (e *engine) loadLocked() error {
	if e.pipeline != 0 {
		return nil
	}
	m := e.models
	errBuf := make([]byte, 512)
	p := e.pipelineLoad(m.dino, m.flow, m.dec, m.slat, m.slatHR, m.shapeDec,
		m.shapeEnc, m.texDec, m.texFlow, m.texFlowHR,
		0 /*flags*/, unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if p == 0 {
		return fmt.Errorf("pipeline load: %s", cstr(errBuf))
	}
	e.stateMu.Lock()
	e.pipeline = p
	e.backend = e.pipelineBackend(p)
	e.caps = int(e.pipelineCaps(p))
	e.textured = e.caps&capTexture != 0
	e.stateMu.Unlock()
	return nil
}

// RunConfig reports the settings the library resolved for the loaded pipeline -
// attention path, chunking gates, CUDA graphs, shape decoder placement - so a
// generation can record what computed it. Several are decided from machine
// state rather than from the request, so the request alone does not explain why
// two runs of one seed differ.
//
// Read from the library rather than from os.Getenv on purpose: the DLL sees the
// CRT's own copy of the environment, which is not necessarily this process's
// view (see the setNativeEnv note in start_server.bat). Returns nil when no
// pipeline is loaded, so a caller records nothing rather than something stale.
func (e *engine) RunConfig() map[string]string {
	e.stateMu.Lock()
	p := e.pipeline
	e.stateMu.Unlock()
	if p == 0 || e.runConfig == nil {
		return nil
	}
	out := map[string]string{}
	for _, line := range strings.Split(e.runConfig(p), "\n") {
		if k, v, ok := strings.Cut(line, "="); ok {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// Unload releases all resident model buffers. Backend/capability metadata is
// retained so /api/info and the quality picker remain useful while idle.
func (e *engine) Unload() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.pipeline == 0 {
		return false
	}
	e.pipelineFree(e.pipeline)
	e.stateMu.Lock()
	e.pipeline = 0
	e.stateMu.Unlock()
	return true
}

func (e *engine) Info() (backend string, caps int, textured, loaded bool) {
	e.stateMu.RLock()
	defer e.stateMu.RUnlock()
	return e.backend, e.caps, e.textured, e.pipeline != 0
}

type meshData struct {
	NVerts  int
	NTris   int
	Verts   []float32 // 3 * NVerts
	Normals []float32 // 3 * NVerts
	Tris    []int32   // 3 * NTris
	PBR     []float32 // 6 * NVerts (base_color rgb, metallic, roughness, alpha); nil if untextured
	// Grid the geometry was extracted at (64/512/1024/1536, or an intermediate
	// multiple of 128 when the cascade token budget stepped 1536 down). 0 for
	// meshes produced by the CPU-side transforms rather than by a generation.
	GridRes int
}

// Generate runs the full image->mesh pipeline. onProgress and onPreview may be
// nil. onPreview receives live intermediate 3D preview blobs (T2VOX01 voxel
// sets) as the sparse structure emerges.
func (e *engine) Generate(image []byte, pipelineType, backgroundMode int, seed uint64, steps int, guidance float32, textureSteps, attentionMode int,
	onLoading func(),
	onProgress func(stage, step, total int),
	onPreview func(stage, step, total int, blob []byte)) (*meshData, error) {

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.pipeline == 0 {
		if onLoading != nil {
			onLoading()
		}
		if err := e.loadLocked(); err != nil {
			return nil, err
		}
	}

	progressMu.Lock()
	progressSink = onProgress
	previewSink = onPreview
	progressMu.Unlock()
	defer func() {
		progressMu.Lock()
		progressSink = nil
		previewSink = nil
		progressMu.Unlock()
	}()

	cb := uintptr(0)
	if onProgress != nil {
		cb = progressCallback
	}
	pv := uintptr(0)
	if onPreview != nil {
		pv = previewCallback
	}

	opts := genOptions{
		Seed:           seed,
		PipelineType:   int32(pipelineType),
		BackgroundMode: int32(backgroundMode),
		Steps:          int32(steps),
		TextureSteps:   int32(textureSteps),
		AttentionMode:  int32(attentionMode),
		Guidance:       guidance,
	}
	errBuf := make([]byte, 512)
	r := e.generate(e.pipeline, unsafe.Pointer(&image[0]), int32(len(image)), &opts,
		cb, nil, pv, nil, unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if r == 0 {
		return nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.meshFree(r)

	nv := int(e.meshNVerts(r))
	nt := int(e.meshNTris(r))
	if nv == 0 || nt == 0 {
		return nil, fmt.Errorf("empty mesh")
	}

	m := &meshData{NVerts: nv, NTris: nt, GridRes: int(e.meshGridRes(r))}
	m.Verts = copyFloats(e.meshVerts(r), 3*nv)
	m.Normals = copyFloats(e.meshNormals(r), 3*nv)
	m.Tris = copyInts(e.meshTris(r), 3*nt)
	if e.meshHasPBR(r) != 0 {
		m.PBR = copyFloats(e.meshPBR(r), 6*nv)
	}
	return m, nil
}

// PrepareMesh returns the exact component-filtered, full-density geometry used by
// GLB export. It is CPU-only and does not require the model pipeline to be loaded.
func (e *engine) PrepareMesh(m *meshData, componentFilter int) (*meshData, error) {
	if m == nil || m.NVerts == 0 || m.NTris == 0 {
		return nil, fmt.Errorf("empty mesh")
	}
	var pbr unsafe.Pointer
	if len(m.PBR) == 6*m.NVerts {
		pbr = unsafe.Pointer(&m.PBR[0])
	}
	errBuf := make([]byte, 512)
	r := e.prepareMeshC(unsafe.Pointer(&m.Verts[0]), int32(m.NVerts),
		unsafe.Pointer(&m.Tris[0]), int32(m.NTris), pbr,
		int32(componentFilter),
		unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if r == 0 {
		return nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.meshFree(r)
	nv, nt := int(e.meshNVerts(r)), int(e.meshNTris(r))
	if nv == 0 || nt == 0 {
		return nil, fmt.Errorf("empty prepared mesh")
	}
	out := &meshData{NVerts: nv, NTris: nt}
	out.Verts = copyFloats(e.meshVerts(r), 3*nv)
	out.Normals = copyFloats(e.meshNormals(r), 3*nv)
	out.Tris = copyInts(e.meshTris(r), 3*nt)
	if e.meshHasPBR(r) != 0 {
		out.PBR = copyFloats(e.meshPBR(r), 6*nv)
	}
	return out, nil
}

// HasPrintRemesh reports whether this library was built with CGAL Alpha Wrap.
func (e *engine) HasPrintRemesh() bool {
	return e != nil && e.printRemeshAvailable != nil && e.printRemeshAvailable() != 0
}

// PreparePrintMesh component-filters and wraps arbitrary source topology in a
// watertight, oriented, intersection-free 2-manifold. Ratios are fractions of
// the source bounding-box diagonal. Alpha Wrap creates new geometry, so when the
// source is textured the material is projected onto the wrap vertices for an
// approximate per-vertex preview; the GLB download rebakes it sharper per texel.
func (e *engine) PreparePrintMesh(m *meshData, componentFilter int, alphaRatio, offsetRatio float32) (*meshData, error) {
	if m == nil || m.NVerts == 0 || m.NTris == 0 {
		return nil, fmt.Errorf("empty mesh")
	}
	if !e.HasPrintRemesh() || e.preparePrintMeshC == nil {
		return nil, fmt.Errorf("print remeshing is unavailable (library was built without CGAL)")
	}
	var pbr unsafe.Pointer
	if len(m.PBR) == 6*m.NVerts {
		pbr = unsafe.Pointer(&m.PBR[0])
	}
	errBuf := make([]byte, 512)
	r := e.preparePrintMeshC(unsafe.Pointer(&m.Verts[0]), int32(m.NVerts),
		unsafe.Pointer(&m.Tris[0]), int32(m.NTris), pbr,
		int32(componentFilter), alphaRatio, offsetRatio,
		unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if r == 0 {
		return nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.meshFree(r)
	nv, nt := int(e.meshNVerts(r)), int(e.meshNTris(r))
	if nv == 0 || nt == 0 {
		return nil, fmt.Errorf("empty print mesh")
	}
	out := &meshData{NVerts: nv, NTris: nt}
	out.Verts = copyFloats(e.meshVerts(r), 3*nv)
	out.Normals = copyFloats(e.meshNormals(r), 3*nv)
	out.Tris = copyInts(e.meshTris(r), 3*nt)
	if e.meshHasPBR(r) != 0 {
		out.PBR = copyFloats(e.meshPBR(r), 6*nv)
	}
	return out, nil
}

// ProjectionBackend names the closest-surface backend behind project_pbr and
// the projected GLB bake: "cgal" or "tinybvh", fixed at build time. Reporting
// only - both produce a result.
func (e *engine) ProjectionBackend() string {
	if e == nil || e.projectionBackendC == nil {
		return ""
	}
	p := e.projectionBackendC()
	if p == 0 {
		return ""
	}
	// Static string owned by the library; read it until the NUL.
	buf := make([]byte, 0, 16)
	for i := 0; i < 64; i++ {
		c := *(*byte)(unsafe.Pointer(p + uintptr(i)))
		if c == 0 {
			break
		}
		buf = append(buf, c)
	}
	return string(buf)
}

// BakeStageTimes is the sub-stage split of the last bake, in seconds. The UV
// unwrap normally dominates; the projection is the part that differs between
// the tinybvh and CGAL backends.
type BakeStageTimes struct {
	Unwrap     float32
	Rasterize  float32
	Projection float32
	TexelFill  float32
	Encode     float32
}

func (e *engine) LastBakeTimings() BakeStageTimes {
	var t BakeStageTimes
	if e == nil || e.lastBakeTimings == nil {
		return t
	}
	e.lastBakeTimings(unsafe.Pointer(&t.Unwrap), unsafe.Pointer(&t.Rasterize),
		unsafe.Pointer(&t.Projection), unsafe.Pointer(&t.TexelFill), unsafe.Pointer(&t.Encode))
	return t
}

// HasQuadRemesh reports whether this library was built with the AutoRemesher
// quad backend. That depends on Eigen 5.x, not on CGAL.
func (e *engine) HasQuadRemesh() bool {
	return e != nil && e.quadRemeshAvailable != nil && e.quadRemeshAvailable() != 0
}

// QuadStats is the quality of a quad remesh, reported so the UI can show it
// instead of implying a guarantee. BoundaryEdges > 0 means the surface is open
// and must not be presented as printable.
type QuadStats struct {
	Quads         int     `json:"quads"`
	Triangles     int     `json:"triangles"`
	NGons         int     `json:"ngons"`
	BoundaryEdges int     `json:"boundary_edges"`
	AreaRetained  float32 `json:"area_retained"`
}

// PrepareQuadMesh rebuilds the mesh as mid-poly quad-dominant topology and
// triangulates it for transport. targetQuads is a density hint, not a face
// count. Unlike PreparePrintMesh this needs no CGAL - and unlike it, the result
// is NOT guaranteed watertight.
func (e *engine) PrepareQuadMesh(m *meshData, componentFilter, targetQuads int, adaptivity float32) (*meshData, *QuadStats, error) {
	if m == nil || m.NVerts == 0 || m.NTris == 0 {
		return nil, nil, fmt.Errorf("empty mesh")
	}
	if !e.HasQuadRemesh() || e.prepareQuadMeshC == nil {
		return nil, nil, fmt.Errorf("quad remeshing is unavailable (library was built without the AutoRemesher backend)")
	}
	var pbr unsafe.Pointer
	if len(m.PBR) == 6*m.NVerts {
		pbr = unsafe.Pointer(&m.PBR[0])
	}
	errBuf := make([]byte, 512)
	r := e.prepareQuadMeshC(unsafe.Pointer(&m.Verts[0]), int32(m.NVerts),
		unsafe.Pointer(&m.Tris[0]), int32(m.NTris), pbr,
		int32(componentFilter), int32(targetQuads), adaptivity,
		unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if r == 0 {
		return nil, nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.meshFree(r)
	nv, nt := int(e.meshNVerts(r)), int(e.meshNTris(r))
	if nv == 0 || nt == 0 {
		return nil, nil, fmt.Errorf("empty quad mesh")
	}
	out := &meshData{NVerts: nv, NTris: nt}
	out.Verts = copyFloats(e.meshVerts(r), 3*nv)
	out.Normals = copyFloats(e.meshNormals(r), 3*nv)
	out.Tris = copyInts(e.meshTris(r), 3*nt)
	if e.meshHasPBR(r) != 0 {
		out.PBR = copyFloats(e.meshPBR(r), 6*nv)
	}

	stats := &QuadStats{}
	if e.quadMeshStats != nil {
		var quads, tris, ngons, boundary int32
		var area float32
		e.quadMeshStats(r, unsafe.Pointer(&quads), unsafe.Pointer(&tris),
			unsafe.Pointer(&ngons), unsafe.Pointer(&boundary), unsafe.Pointer(&area))
		stats.Quads, stats.Triangles, stats.NGons = int(quads), int(tris), int(ngons)
		stats.BoundaryEdges, stats.AreaRetained = int(boundary), area
	}
	return out, stats, nil
}

// BakeGLB turns a generated mesh into a portable vertex-coloured GLB. texSize
// remains an atlas hint for the explicit T2GLB_XATLAS mode. Geometry retains its
// original polygon density. CPU-only; it can run while the GPU is idle.
func (e *engine) BakeGLB(m *meshData, texSize, componentFilter int, mat glbMaterial) ([]byte, error) {
	if m == nil || m.NVerts == 0 || m.NTris == 0 {
		return nil, fmt.Errorf("empty mesh")
	}
	var pbr unsafe.Pointer
	if len(m.PBR) == 6*m.NVerts {
		pbr = unsafe.Pointer(&m.PBR[0])
	}
	opts := glbOptions{TextureSize: int32(texSize), ComponentFilter: int32(componentFilter),
		NormalMap: 1, ExportFlags: mat.flags(), UnitScale: mat.UnitScale}
	var outLen int32
	errBuf := make([]byte, 512)
	p := e.bakeGLB(unsafe.Pointer(&m.Verts[0]), int32(m.NVerts),
		unsafe.Pointer(&m.Tris[0]), int32(m.NTris), pbr,
		unsafe.Pointer(&opts),
		unsafe.Pointer(&outLen), unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if p == 0 {
		return nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.freeBuffer(p)
	out := make([]byte, outLen)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(p)), int(outLen)))
	return out, nil
}

// NormalMapStats is how much of a normal map bake the closest-surface search
// had to leave flat. A nearest-point transfer is not a bake cage: on thin
// geometry it can land on the far side of a wall, and baking that would invert
// the shading lobe over a visible patch. Those texels stay flat and are counted
// here so the UI can show the loss rather than implying full detail transfer.
type NormalMapStats struct {
	CoveredTexels  int32
	RejectedTexels int32
}

func (e *engine) LastNormalMapStats() NormalMapStats {
	var s NormalMapStats
	if e == nil || e.lastNormalMapStats == nil {
		return s
	}
	e.lastNormalMapStats(unsafe.Pointer(&s.CoveredTexels), unsafe.Pointer(&s.RejectedTexels))
	return s
}

// BakeProjectedGLB UV-unwraps replacement geometry and transfers the dense
// source material per texel through the portable closest-surface backend. With
// normalMap set it additionally bakes a tangent-space normal map and exports
// the MikkTSpace-compatible TANGENT attribute it was baked against, which is
// what carries the dense mesh's surface detail onto the replacement geometry.
func (e *engine) BakeProjectedGLB(target, source *meshData, texSize, sourceComponentFilter int, normalMap bool, mat glbMaterial) ([]byte, error) {
	if target == nil || target.NVerts == 0 || target.NTris == 0 ||
		source == nil || source.NVerts == 0 || source.NTris == 0 || len(source.PBR) != 6*source.NVerts {
		return nil, fmt.Errorf("empty projected GLB mesh or missing source PBR")
	}
	// Deliberately not gated on HasPrintRemesh: the closest-surface projection
	// moved to tinybvh and works in every build. Only alpha_wrap still needs
	// CGAL, and that is PreparePrintMesh's problem, not this call's.
	if e.bakeProjectedGLB == nil {
		return nil, fmt.Errorf("PBR projection is unavailable")
	}
	var outLen int32
	normalMapFlag := int32(0)
	if normalMap {
		normalMapFlag = 1
	}
	opts := glbOptions{TextureSize: int32(texSize), ComponentFilter: int32(sourceComponentFilter),
		NormalMap: normalMapFlag, ExportFlags: mat.flags(), UnitScale: mat.UnitScale}
	errBuf := make([]byte, 512)
	p := e.bakeProjectedGLB(
		unsafe.Pointer(&target.Verts[0]), int32(target.NVerts),
		unsafe.Pointer(&target.Tris[0]), int32(target.NTris),
		unsafe.Pointer(&source.Verts[0]), int32(source.NVerts),
		unsafe.Pointer(&source.Tris[0]), int32(source.NTris),
		unsafe.Pointer(&source.PBR[0]), unsafe.Pointer(&opts),
		unsafe.Pointer(&outLen), unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if p == 0 {
		return nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.freeBuffer(p)
	out := make([]byte, outLen)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(p)), int(outLen)))
	return out, nil
}

func copyFloats(p uintptr, n int) []float32 {
	src := unsafe.Slice((*float32)(unsafe.Pointer(p)), n)
	dst := make([]float32, n)
	copy(dst, src)
	return dst
}

func copyInts(p uintptr, n int) []int32 {
	src := unsafe.Slice((*int32)(unsafe.Pointer(p)), n)
	dst := make([]int32, n)
	copy(dst, src)
	return dst
}

func cstr(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
