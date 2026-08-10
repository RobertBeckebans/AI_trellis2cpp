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
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

const abiVersion = 11

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

	capCoarse  = 1
	cap512     = 2
	cap1024    = 4
	capTexture = 8

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
	generate        func(p uintptr, img unsafe.Pointer, imgLen int32, pipelineType, backgroundMode int32,
		seed uint64, steps int32, guidance float32, textureSteps int32, cb uintptr, user unsafe.Pointer,
		preview uintptr, previewUser unsafe.Pointer,
		err unsafe.Pointer, errLen int32) uintptr
	meshNVerts   func(r uintptr) int32
	meshNTris    func(r uintptr) int32
	meshVerts    func(r uintptr) uintptr
	meshNormals  func(r uintptr) uintptr
	meshTris     func(r uintptr) uintptr
	meshHasPBR   func(r uintptr) int32
	meshPBR      func(r uintptr) uintptr
	meshFree     func(r uintptr)
	prepareMeshC func(verts unsafe.Pointer, nv int32, tris unsafe.Pointer, nt int32, pbr unsafe.Pointer,
		componentFilter int32, err unsafe.Pointer, errLen int32) uintptr
	printRemeshAvailable func() int32
	preparePrintMeshC    func(verts unsafe.Pointer, nv int32, tris unsafe.Pointer, nt int32, pbr unsafe.Pointer,
		componentFilter int32, alphaRatio, offsetRatio float32,
		err unsafe.Pointer, errLen int32) uintptr
	bakeGLB func(verts unsafe.Pointer, nv int32, tris unsafe.Pointer, nt int32, pbr unsafe.Pointer,
		texSize, componentFilter int32, outLen unsafe.Pointer, err unsafe.Pointer, errLen int32) uintptr
	bakeProjectedGLB func(targetVerts unsafe.Pointer, targetNV int32, targetTris unsafe.Pointer, targetNT int32,
		sourceVerts unsafe.Pointer, sourceNV int32, sourceTris unsafe.Pointer, sourceNT int32,
		sourcePBR unsafe.Pointer, texSize, sourceComponentFilter int32,
		outLen unsafe.Pointer, err unsafe.Pointer, errLen int32) uintptr
	freeBuffer func(buf uintptr)
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
	purego.RegisterLibFunc(&e.generate, lib, "t2_generate")
	purego.RegisterLibFunc(&e.meshNVerts, lib, "t2_mesh_n_verts")
	purego.RegisterLibFunc(&e.meshNTris, lib, "t2_mesh_n_tris")
	purego.RegisterLibFunc(&e.meshVerts, lib, "t2_mesh_verts")
	purego.RegisterLibFunc(&e.meshNormals, lib, "t2_mesh_normals")
	purego.RegisterLibFunc(&e.meshTris, lib, "t2_mesh_tris")
	purego.RegisterLibFunc(&e.meshHasPBR, lib, "t2_mesh_has_pbr")
	purego.RegisterLibFunc(&e.meshPBR, lib, "t2_mesh_pbr")
	purego.RegisterLibFunc(&e.meshFree, lib, "t2_mesh_free")
	purego.RegisterLibFunc(&e.prepareMeshC, lib, "t2_prepare_mesh")
	purego.RegisterLibFunc(&e.printRemeshAvailable, lib, "t2_print_remesh_available")
	purego.RegisterLibFunc(&e.preparePrintMeshC, lib, "t2_prepare_print_mesh")
	purego.RegisterLibFunc(&e.bakeGLB, lib, "t2_bake_glb")
	purego.RegisterLibFunc(&e.bakeProjectedGLB, lib, "t2_bake_projected_glb")
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
			caps |= cap1024
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
}

// Generate runs the full image->mesh pipeline. onProgress and onPreview may be
// nil. onPreview receives live intermediate 3D preview blobs (T2VOX01 voxel
// sets) as the sparse structure emerges.
func (e *engine) Generate(image []byte, pipelineType, backgroundMode int, seed uint64, steps int, guidance float32, textureSteps int,
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

	errBuf := make([]byte, 512)
	r := e.generate(e.pipeline, unsafe.Pointer(&image[0]), int32(len(image)), int32(pipelineType),
		int32(backgroundMode), seed, int32(steps), guidance, int32(textureSteps), cb, nil, pv, nil,
		unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if r == 0 {
		return nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.meshFree(r)

	nv := int(e.meshNVerts(r))
	nt := int(e.meshNTris(r))
	if nv == 0 || nt == 0 {
		return nil, fmt.Errorf("empty mesh")
	}

	m := &meshData{NVerts: nv, NTris: nt}
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

// BakeGLB turns a generated mesh into a portable vertex-coloured GLB. texSize
// remains an atlas hint for the explicit T2GLB_XATLAS mode. Geometry retains its
// original polygon density. CPU-only; it can run while the GPU is idle.
func (e *engine) BakeGLB(m *meshData, texSize, componentFilter int) ([]byte, error) {
	if m == nil || m.NVerts == 0 || m.NTris == 0 {
		return nil, fmt.Errorf("empty mesh")
	}
	var pbr unsafe.Pointer
	if len(m.PBR) == 6*m.NVerts {
		pbr = unsafe.Pointer(&m.PBR[0])
	}
	var outLen int32
	errBuf := make([]byte, 512)
	p := e.bakeGLB(unsafe.Pointer(&m.Verts[0]), int32(m.NVerts),
		unsafe.Pointer(&m.Tris[0]), int32(m.NTris), pbr,
		int32(texSize), int32(componentFilter),
		unsafe.Pointer(&outLen), unsafe.Pointer(&errBuf[0]), int32(len(errBuf)))
	if p == 0 {
		return nil, fmt.Errorf("%s", cstr(errBuf))
	}
	defer e.freeBuffer(p)
	out := make([]byte, outLen)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(p)), int(outLen)))
	return out, nil
}

// BakeProjectedGLB UV-unwraps replacement geometry and transfers the dense
// source material per texel through the portable CGAL closest-surface backend.
func (e *engine) BakeProjectedGLB(target, source *meshData, texSize, sourceComponentFilter int) ([]byte, error) {
	if target == nil || target.NVerts == 0 || target.NTris == 0 ||
		source == nil || source.NVerts == 0 || source.NTris == 0 || len(source.PBR) != 6*source.NVerts {
		return nil, fmt.Errorf("empty projected GLB mesh or missing source PBR")
	}
	if !e.HasPrintRemesh() || e.bakeProjectedGLB == nil {
		return nil, fmt.Errorf("PBR projection is unavailable (library was built without CGAL)")
	}
	var outLen int32
	errBuf := make([]byte, 512)
	p := e.bakeProjectedGLB(
		unsafe.Pointer(&target.Verts[0]), int32(target.NVerts),
		unsafe.Pointer(&target.Tris[0]), int32(target.NTris),
		unsafe.Pointer(&source.Verts[0]), int32(source.NVerts),
		unsafe.Pointer(&source.Tris[0]), int32(source.NTris),
		unsafe.Pointer(&source.PBR[0]), int32(texSize), int32(sourceComponentFilter),
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
