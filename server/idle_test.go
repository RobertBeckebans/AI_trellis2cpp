package main

import (
	"net/http/httptest"
	"testing"
	"unsafe"
)

func testEngine(pipeline uintptr, freed *int) *engine {
	return &engine{
		pipeline: pipeline,
		backend:  "test GPU",
		caps:     capCoarse | cap512 | capTexture,
		textured: true,
		pipelineFree: func(uintptr) {
			*freed++
		},
	}
}

func TestEngineUnloadRetainsRuntimeInfoAndReloads(t *testing.T) {
	freed, loads := 0, 0
	e := testEngine(41, &freed)
	e.models.dino = "dino.gguf"
	e.pipelineLoad = func(dino, flow, dec, slat, slatHR, shapeDec, shapeEnc, texDec, texFlow, texFlowHR string,
		flags int32, err unsafe.Pointer, errLen int32) uintptr {
		loads++
		if dino != "dino.gguf" {
			t.Fatalf("reload used dino path %q", dino)
		}
		return 42
	}
	e.pipelineBackend = func(uintptr) string { return "test GPU" }
	e.pipelineCaps = func(uintptr) int32 { return capCoarse | cap512 | capTexture }

	if !e.Unload() || freed != 1 {
		t.Fatalf("Unload() = true with one free wanted; freed=%d", freed)
	}
	backend, caps, textured, loaded := e.Info()
	if loaded || backend != "test GPU" || caps != capCoarse|cap512|capTexture || !textured {
		t.Fatalf("Info() after unload = %q, %d, %v, %v", backend, caps, textured, loaded)
	}

	e.mu.Lock()
	err := e.loadLocked()
	e.mu.Unlock()
	if err != nil || loads != 1 {
		t.Fatalf("reload: err=%v loads=%d", err, loads)
	}
	_, _, _, loaded = e.Info()
	if !loaded {
		t.Fatal("pipeline is not marked loaded after reload")
	}
}

func TestConfiguredCapsForLazyStartup(t *testing.T) {
	tests := []struct {
		name string
		m    engineModels
		want int
	}{
		{"coarse", engineModels{}, capCoarse},
		{"512", engineModels{slat: "slat", shapeDec: "shape"}, capCoarse | cap512},
		{"1024 textured", engineModels{
			slat: "slat", slatHR: "hr", shapeDec: "shape", shapeEnc: "shapeenc",
			texDec: "texdec", texFlow: "texflow",
		}, capCoarse | cap512 | cap1024 | capTexture},
		{"texture missing encoder", engineModels{
			slat: "slat", shapeDec: "shape", texDec: "texdec", texFlow: "texflow",
		}, capCoarse | cap512},
		{"incomplete fine", engineModels{slat: "slat", texDec: "texdec", texFlow: "texflow"}, capCoarse},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := configuredCaps(tt.m); got != tt.want {
				t.Fatalf("configuredCaps() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestIdlePolicyWaitsForQueuedWork(t *testing.T) {
	freed := 0
	e := testEngine(41, &freed)
	s := &server{eng: e, jobs: map[string]*job{}, q: make(chan *job, 1), queued: 1}

	if s.setUnloadIdle(true) {
		t.Fatal("enabled idle policy unloaded with queued work")
	}
	if freed != 0 {
		t.Fatalf("pipeline freed with queued work: %d", freed)
	}

	s.mu.Lock()
	s.queued = 0
	s.mu.Unlock()
	if !s.unloadModelsIfIdle(nil) || freed != 1 {
		t.Fatalf("idle unload failed; freed=%d", freed)
	}
}

func TestParseExportOptions(t *testing.T) {
	def := parseExportOptions(httptest.NewRequest("GET", "/api/glb/job", nil))
	if def.componentFilter != 2 || def.printWrap || def.prepareKey() != "2" {
		t.Fatalf("default export should preserve all components: %+v", def)
	}
	r := httptest.NewRequest("GET", "/api/glb/job?tex=1024&components=largest", nil)
	o := parseExportOptions(r)
	if o.textureSize != 1024 || o.componentFilter != 1 {
		t.Fatalf("parseExportOptions() = %+v", o)
	}
	if o.prepareKey() != "1" || o.glbKey() != "1024-1" {
		t.Fatalf("unexpected export cache keys: %q %q", o.prepareKey(), o.glbKey())
	}
	wrapped := parseExportOptions(httptest.NewRequest("GET",
		"/api/glb/job?print=1&alpha=2.5&offset=0.1", nil))
	if !wrapped.printWrap || wrapped.alphaRatio != 0.025 || wrapped.offsetRatio != 0.001 ||
		wrapped.prepareKey() == def.prepareKey() {
		t.Fatalf("print-wrap options were not parsed/cached independently: %+v", wrapped)
	}
}

func TestKeepAllExportPreviewUsesOriginalMesh(t *testing.T) {
	original := &meshData{NVerts: 3, NTris: 1}
	j := &job{mesh: original}
	s := &server{}
	got, err := s.preparedExportMesh(j, exportOptions{textureSize: 2048, componentFilter: 2}, newStageLog())
	if err != nil {
		t.Fatal(err)
	}
	if got != original {
		t.Fatal("keep-all export preview did not return the exact source mesh")
	}
}
