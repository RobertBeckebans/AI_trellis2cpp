package main

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func testPersistedMesh(textured bool) *meshData {
	m := &meshData{
		NVerts: 3, NTris: 1,
		Verts:   []float32{0, 0, 0, 1, 0, 0, 0, 1, 0},
		Normals: []float32{0, 0, 1, 0, 0, 1, 0, 0, 1},
		Tris:    []int32{0, 1, 2},
	}
	if textured {
		m.PBR = []float32{
			1, 0, 0, 0.1, 0.2, 1,
			0, 1, 0, 0.3, 0.4, 0.9,
			0, 0, 1, 0.5, 0.6, 0.8,
		}
	}
	return m
}

func TestMeshFileRoundTrip(t *testing.T) {
	for _, textured := range []bool{false, true} {
		t.Run(map[bool]string{false: "geometry", true: "pbr"}[textured], func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "mesh.t2mesh")
			want := testPersistedMesh(textured)
			if err := writeMeshFile(path, want); err != nil {
				t.Fatal(err)
			}
			got, err := readMeshFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("mesh round trip mismatch\n got: %#v\nwant: %#v", got, want)
			}
		})
	}
}

func TestPersistAndRestoreCompletedJob(t *testing.T) {
	store := t.TempDir()
	s := &server{jobs: map[string]*job{}, storeDir: store}
	wantFrames := []frameMeta{
		{Stage: "sampling sparse structure", Step: 1, Total: 2, Kind: "voxel"},
		{Stage: "sampling shape SLAT", Step: 2, Total: 2, Kind: "mesh"},
	}
	wantPreviews := [][]byte{[]byte("T2VOX01-frame"), []byte("T2MESH01-frame")}
	j := &job{
		ID: "0123456789abcdef", State: "running", CreatedAt: 123456789,
		StartedAt: 123456800, FinishedAt: 123499000, DurationMS: 42200,
		Quality: "1024", Thumbnail: "data:image/jpeg;base64,dGVzdA==",
		PreviewSeq: len(wantFrames), Frames: wantFrames, LivePreview: true,
		StageTimings: []stageTiming{{Stage: "sampling sparse structure", Milliseconds: 21000}},
		pipeline:     pipe1024, Seed: 42, Steps: 12, TextureSteps: 10, Guidance: 7.5,
		previews: wantPreviews, mesh: testPersistedMesh(true),
		image: []byte("processed-input"), source: []byte("exact-original"),
	}
	if err := s.persistJob(j); err != nil {
		t.Fatal(err)
	}
	if len(j.previews) != 0 || j.persistDir == "" || j.meshPath == "" ||
		j.inputPath == "" || j.sourcePath == "" {
		t.Fatalf("persist did not switch assets to disk: previews=%d dir=%q mesh=%q input=%q source=%q",
			len(j.previews), j.persistDir, j.meshPath, j.inputPath, j.sourcePath)
	}

	restarted := &server{jobs: map[string]*job{}, storeDir: store}
	n, err := restarted.restoreJobs()
	if err != nil || n != 1 {
		t.Fatalf("restoreJobs() = %d, %v; want 1, nil", n, err)
	}
	got := restarted.jobs[j.ID]
	if got == nil || got.State != "done" || got.Quality != "1024" ||
		got.Thumbnail != j.Thumbnail || got.CreatedAt != j.CreatedAt ||
		got.StartedAt != j.StartedAt || got.FinishedAt != j.FinishedAt ||
		got.DurationMS != j.DurationMS || got.LivePreview != j.LivePreview ||
		!reflect.DeepEqual(got.StageTimings, j.StageTimings) {
		t.Fatalf("restored metadata = %#v", got)
	}
	if got.mesh != nil || len(got.previews) != 0 {
		t.Fatal("restored binary assets should remain lazy until requested")
	}
	if source, err := os.ReadFile(got.sourcePath); err != nil || string(source) != "exact-original" {
		t.Fatalf("restored source image = %q, %v", source, err)
	}
	if input, err := os.ReadFile(got.inputPath); err != nil || string(input) != "processed-input" {
		t.Fatalf("restored generation input = %q, %v", input, err)
	}
	rr := httptest.NewRecorder()
	restarted.handleSource(rr, httptest.NewRequest(http.MethodGet, "/api/source/"+j.ID, nil))
	if rr.Code != http.StatusOK || rr.Body.String() != "exact-original" {
		t.Fatalf("GET source = %d %q", rr.Code, rr.Body.String())
	}
	mesh, err := restarted.loadJobMesh(got)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mesh, j.mesh) {
		t.Fatal("restored mesh mismatch")
	}
	for i, want := range wantPreviews {
		blob, err := loadJobPreview(got, i)
		if err != nil {
			t.Fatalf("preview %d: %v", i, err)
		}
		if !bytes.Equal(blob, want) {
			t.Fatalf("preview %d = %q, want %q", i, blob, want)
		}
	}
}

func TestRegenerateUsesPersistedInputWithoutUpload(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "input.img")
	sourcePath := filepath.Join(dir, "source.img")
	if err := os.WriteFile(inputPath, []byte("processed-input"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sourcePath, []byte("exact-original"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := &job{
		ID: "old", State: "done", CreatedAt: 1, Quality: "512", Thumbnail: "thumb",
		inputPath: inputPath, sourcePath: sourcePath, pipeline: pipe512,
		Seed: 7, Steps: 8, TextureSteps: 9, Guidance: 6.5,
	}
	s := &server{jobs: map[string]*job{"old": old}, q: make(chan *job, 1)}
	req := httptest.NewRequest(http.MethodPost, "/api/regenerate/old",
		strings.NewReader("quality=1024&seed=42&steps=14&texture_steps=15&guidance=8&preview=0"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rr := httptest.NewRecorder()
	s.handleRegenerate(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("POST regenerate = %d: %s", rr.Code, rr.Body.String())
	}
	var response map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	regenerated := s.jobs[response["job"]]
	if regenerated == nil {
		t.Fatal("regenerated job was not queued")
	}
	if string(regenerated.image) != "processed-input" || string(regenerated.source) != "exact-original" {
		t.Fatalf("regenerated bytes = input %q, source %q", regenerated.image, regenerated.source)
	}
	if regenerated.Quality != "1024" || regenerated.pipeline != pipe1024 ||
		regenerated.Background != backgroundKeep || regenerated.Seed != 42 ||
		regenerated.Steps != 14 || regenerated.TextureSteps != 15 ||
		regenerated.Guidance != 8 || regenerated.LivePreview {
		t.Fatalf("regenerated settings = %#v", regenerated)
	}
}

func TestGenerateKeepsOriginalSeparateFromProcessedInput(t *testing.T) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	inputPart, err := mw.CreateFormFile("image", "input.png")
	if err != nil {
		t.Fatal(err)
	}
	inputPart.Write([]byte("processed-png"))
	sourcePart, err := mw.CreateFormFile("source", "camera-original.webp")
	if err != nil {
		t.Fatal(err)
	}
	sourcePart.Write([]byte("exact-original-webp"))
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}

	s := &server{jobs: map[string]*job{}, q: make(chan *job, 1)}
	req := httptest.NewRequest(http.MethodPost, "/api/generate", &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rr := httptest.NewRecorder()
	s.handleGenerate(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("POST generate = %d: %s", rr.Code, rr.Body.String())
	}
	var response map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	j := s.jobs[response["job"]]
	if j == nil || string(j.image) != "processed-png" || string(j.source) != "exact-original-webp" {
		t.Fatalf("queued job bytes = %#v", j)
	}
}

func TestServerHistoryListAndDelete(t *testing.T) {
	root := t.TempDir()
	oldDir := filepath.Join(root, "old")
	if err := os.Mkdir(oldDir, 0o755); err != nil {
		t.Fatal(err)
	}
	old := &job{ID: "old", State: "done", CreatedAt: 10, Quality: "512", persistDir: oldDir}
	newer := &job{ID: "new", State: "done", CreatedAt: 20, Quality: "1024", Thumbnail: "thumb"}
	active := &job{ID: "active", State: "running", CreatedAt: 30}
	s := &server{jobs: map[string]*job{"old": old, "new": newer, "active": active}}

	rr := httptest.NewRecorder()
	s.handleJobs(rr, httptest.NewRequest(http.MethodGet, "/api/jobs", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("GET /api/jobs = %d: %s", rr.Code, rr.Body.String())
	}
	var got []jobSummary
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].ID != "new" || got[1].ID != "old" {
		t.Fatalf("history order/content = %#v", got)
	}

	rr = httptest.NewRecorder()
	s.handleJob(rr, httptest.NewRequest(http.MethodDelete, "/api/job/old", nil))
	if rr.Code != http.StatusNoContent {
		t.Fatalf("DELETE /api/job/old = %d: %s", rr.Code, rr.Body.String())
	}
	if s.jobs["old"] != nil {
		t.Fatal("deleted job remains in server index")
	}
	if _, err := os.Stat(oldDir); !os.IsNotExist(err) {
		t.Fatalf("deleted job directory still exists: %v", err)
	}
}

func TestRestoreIgnoresIncompleteTemporaryDirectory(t *testing.T) {
	store := t.TempDir()
	if err := os.Mkdir(filepath.Join(store, ".unfinished.tmp-123"), 0o755); err != nil {
		t.Fatal(err)
	}
	s := &server{jobs: map[string]*job{}, storeDir: store}
	n, err := s.restoreJobs()
	if err != nil || n != 0 {
		t.Fatalf("restoreJobs() = %d, %v; want 0, nil", n, err)
	}
}
