// trellis2 demo server: upload an image, get a 3D mesh back.
//
//	cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
//	cd server && CGO_ENABLED=0 go build -o trellis2-server .
//	./trellis2-server -lib ../build/libtrellis2.so -ggufs ../ggufs
//
// API:
//
//	GET  /                 self-contained WebGL viewer (embedded web/index.html)
//	GET  /api/info         {backend, defaults}
//	POST /api/settings     form unload_idle=0|1 -> current runtime settings
//	POST /api/generate     multipart image [+ source, background, seed, steps, guidance, texture_steps, preview=1] -> {job}
//	POST /api/regenerate/{id} settings only; reuse the persisted generation input -> {job}
//	GET  /api/jobs         durable completed-generation history
//	GET  /api/job/{id}     {state, stage, step, total, previewSeq, livePreview, durationMs, stageTimings, error}
//	GET  /api/source/{id}  original uploaded image (byte-for-byte when supplied)
//	GET  /api/mesh/{id}    binary mesh: T2MESH01 geometry or T2MESH03 geometry +
//	                       f32[6nv] PBR (base RGB, metal, roughness, alpha)
//	GET  /api/export-preview/{id} full-density/component-filtered T2MESH0* geometry
//	GET  /api/preview/{id} latest live 3D preview: "T2VOX01" u32 res u32 nvox
//	                       u16[3nvox] voxel coords (little-endian)
package main

import (
	"bytes"
	"embed"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed web
var webFS embed.FS

const maxImage = 32 << 20                    // 32 MiB per original/processed image
const maxUpload = (2 * maxImage) + (2 << 20) // both images plus multipart fields
const maxFrames = 256                        // cap recorded preview frames per job (bounds memory)

// frameMeta describes one recorded intermediate-preview frame so the viewer can
// label and order the scrubber. Kind is "voxel" (T2VOX01) or "mesh" (T2MESH0*).
type frameMeta struct {
	Stage string `json:"stage"`
	Step  int    `json:"step"`
	Total int    `json:"total"`
	Kind  string `json:"kind"`
}

// stageTiming records aggregate wall time spent in a reported pipeline stage.
// It includes CPU work and host/device handoff time, which a GPU-utilisation
// graph alone cannot explain.
// stageLog accumulates named phase durations and prints them as one block at
// the end, so a slow run can be attributed to a phase instead of guessed at.
// Repeated names accumulate, which keeps a per-step stage on one line.
type stageLog struct {
	mu      sync.Mutex
	order   []string
	ms      map[string]int64
	started time.Time
}

func newStageLog() *stageLog {
	return &stageLog{ms: map[string]int64{}, started: time.Now()}
}

func (sl *stageLog) add(name string, d time.Duration) {
	if sl == nil {
		return
	}
	sl.mu.Lock()
	defer sl.mu.Unlock()
	if _, seen := sl.ms[name]; !seen {
		sl.order = append(sl.order, name)
	}
	sl.ms[name] += d.Milliseconds()
}

// track runs fn and books its duration under name.
func (sl *stageLog) track(name string, fn func() error) error {
	started := time.Now()
	err := fn()
	sl.add(name, time.Since(started))
	return err
}

// dump prints the block. Nothing is printed when no phase was recorded, so a
// cache hit stays quiet instead of logging an empty table.
func (sl *stageLog) dump(title string) {
	if sl == nil {
		return
	}
	sl.mu.Lock()
	defer sl.mu.Unlock()
	if len(sl.order) == 0 {
		return
	}
	log.Printf("─── %s ───", title)
	for _, name := range sl.order {
		// Names starting with "-" are sub-stages of the entry above them.
		if strings.HasPrefix(name, "-") {
			log.Printf("    %-30s %6.2f s", name, float64(sl.ms[name])/1000)
			continue
		}
		log.Printf("  %-32s %6.2f s", name, float64(sl.ms[name])/1000)
	}
	log.Printf("  %-32s %6.2f s", "total", time.Since(sl.started).Seconds())
}

// track runs fn under a heartbeat: none of the export stages report progress
// across the C ABI, and on real geometry each of them can run for minutes. The
// periodic line is the difference between "working" and "hung" for whoever is
// watching the console.
func (sl *stageLog) trackLive(name, verb string, fn func() error) error {
	done := make(chan struct{})
	go func() {
		started := time.Now()
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				log.Printf("    ... still %s (%.0f s)", verb, time.Since(started).Seconds())
			}
		}
	}()
	err := sl.track(name, fn)
	close(done)
	return err
}

type stageTiming struct {
	Stage        string `json:"stage"`
	Milliseconds int64  `json:"milliseconds"`
}

type job struct {
	mu    sync.Mutex
	ID    string `json:"id"`
	State string `json:"state"` // queued | running | done | error
	// CreatedAt and Quality are retained in the durable manifest for future
	// server-side history/indexing as well as diagnostics.
	CreatedAt  int64  `json:"createdAt,omitempty"`
	StartedAt  int64  `json:"startedAt,omitempty"`
	FinishedAt int64  `json:"finishedAt,omitempty"`
	DurationMS int64  `json:"durationMs,omitempty"`
	Quality    string `json:"quality,omitempty"`
	// Grid the geometry was actually extracted at. Usually the tier Quality
	// asked for, but the 1536 cascade's token budget can step it back toward
	// 1024 in units of 128 on a dense object — that has to be visible rather
	// than showing up as a mesh that is quietly coarser than requested.
	Resolution int `json:"resolution,omitempty"`
	// Backend the models were actually loaded on, as ggml reports it — e.g.
	// "AMD Radeon AI PRO R9700 [Vulkan0]". Recorded per generation because a
	// build carrying both GPU backends can switch between them at runtime, and
	// two meshes from the same seed then differ for a reason that is otherwise
	// invisible afterwards. The requested setting is not enough: "auto" says
	// nothing about what it resolved to.
	Device     string `json:"device,omitempty"`
	Thumbnail  string `json:"thumbnail,omitempty"`
	Stage      string `json:"stage,omitempty"`
	Step       int    `json:"step"`
	Total      int    `json:"total"`
	Error      string `json:"error,omitempty"`

	// Recorded intermediate 3D previews: every frame is kept (not just the
	// latest) so the viewer can scrub the whole generation back and forth.
	// PreviewSeq == number of frames recorded; the browser fetches any frame it
	// is still missing by index (/api/preview/{id}?seq=N). Frames carries the
	// per-frame stage/step/kind metadata for labelling the scrubber.
	PreviewSeq   int           `json:"previewSeq"`
	PreviewStage string        `json:"previewStage,omitempty"`
	PreviewStep  int           `json:"previewStep"`
	PreviewTotal int           `json:"previewTotal"`
	Frames       []frameMeta   `json:"frames,omitempty"`
	LivePreview  bool          `json:"livePreview"`
	StageTimings []stageTiming `json:"stageTimings,omitempty"`

	// The sampling parameters, exported so a client can read back exactly what
	// produced a stored generation and reproduce it. `shapeSteps` rather than
	// `steps` because `step`/`total` above are the live progress counter, and
	// two similarly named fields in one object invite the wrong one being read.
	Seed         uint64  `json:"seed"`
	Guidance     float32 `json:"guidance"`
	Steps        int     `json:"shapeSteps"`
	TextureSteps int     `json:"textureSteps"`
	Background   int     `json:"background"`

	image        []byte // processed image passed to TRELLIS
	source       []byte // exact original upload, used for display and future edits
	pipeline     int
	keyframes    int      // intermediate shape-SLAT mesh keyframes to record (0 = off)
	previews     [][]byte // every preview blob, in order; served by /api/preview?seq=
	mesh         *meshData
	exportMesh   *meshData  // cached component-cleanup/print-wrap preview
	exportKey    string     // component mode + optional Alpha Wrap / quad parameters
	quadStats    *QuadStats // quality of the last quad remesh, nil when unused
	// How much of the last normal map bake the closest-surface search had to
	// leave flat. nil when no normal map was baked.
	normalMapStats *NormalMapStats
	glb          []byte     // cached last GLB bake
	glbKey       string     // "tex-components" the cached GLB was baked with
	exportMu     sync.Mutex // serializes large export preparation/bakes per job
	persistDir   string     // committed on-disk job directory; empty before save
	meshPath     string     // final T2MESH file, loaded lazily after a restart
	inputPath    string     // processed generation input, persisted for regeneration
	sourcePath   string     // exact original upload, persisted for showcase/display
}

type server struct {
	eng        *engine
	mu         sync.Mutex
	lifecycle  sync.Mutex // serializes idle unloads with accepting new work
	jobs       map[string]*job
	q          chan *job
	queued     int
	active     bool
	unloadIdle bool
	storeDir   string // durable completed-job store; empty disables persistence
}

func (s *server) worker() {
	for j := range s.q {
		started := time.Now()
		currentStage := ""
		stageStarted := started
		addStageTime := func(stage string, elapsed time.Duration) {
			ms := elapsed.Milliseconds()
			if ms < 0 {
				return
			}
			for i := range j.StageTimings {
				if j.StageTimings[i].Stage == stage {
					j.StageTimings[i].Milliseconds += ms
					return
				}
			}
			j.StageTimings = append(j.StageTimings, stageTiming{Stage: stage, Milliseconds: ms})
		}
		lastProgressLog := time.Time{}
		setStage := func(stage string, step, total int) {
			now := time.Now()
			j.mu.Lock()
			if stage != currentStage {
				if currentStage != "" {
					addStageTime(currentStage, now.Sub(stageStarted))
				}
				currentStage, stageStarted = stage, now
				lastProgressLog = time.Time{}
				log.Printf("  %s ...", stage)
			}
			j.Stage, j.Step, j.Total = stage, step, total
			j.mu.Unlock()
			// Minimal progress trace, throttled: a multi-minute stage should look
			// alive in the console rather than hung. Only stages reporting a step
			// count produce these — the upsample stage's step/total are the
			// cascade's (achieved, requested) resolution, already logged above,
			// and would otherwise read as a nonsensical "1536 / 1536".
			if total > 0 && stage != stageNames[stageUpsample] &&
				(now.Sub(lastProgressLog) >= 2*time.Second || step >= total) {
				lastProgressLog = now
				log.Printf("    ... %d / %d (%.1f s)", step, total, now.Sub(stageStarted).Seconds())
			}
		}
		finishTiming := func() {
			finished := time.Now()
			j.mu.Lock()
			if currentStage != "" {
				addStageTime(currentStage, finished.Sub(stageStarted))
				currentStage = ""
			}
			j.FinishedAt = finished.UnixMilli()
			j.DurationMS = finished.Sub(started).Milliseconds()
			j.mu.Unlock()
		}

		s.mu.Lock()
		s.queued--
		s.active = true
		s.mu.Unlock()

		j.mu.Lock()
		j.State = "running"
		j.StartedAt = started.UnixMilli()
		j.mu.Unlock()

		// Everything that decides what this run produces, on one line before it
		// starts. Two runs are only comparable if the parameters were the same,
		// and reconstructing them from the browser afterwards does not scale
		// past a couple of jobs.
		quality := j.Quality
		if quality == "" {
			quality = "auto"
		}
		// Live preview is already on the "finished" line, so it is left out here.
		log.Printf("job %s: quality %s, seed %d, guidance %.2f, shape steps %d, texture steps %d, background %d",
			j.ID, quality, j.Seed, j.Guidance, j.Steps, j.TextureSteps, j.Background)

		// The library reads T2_KEYFRAMES for opt-in intermediate mesh keyframes.
		// A single worker goroutine runs generations serially, so setting the env
		// per-job here is race-free.
		os.Setenv("T2_KEYFRAMES", strconv.Itoa(j.keyframes))

		var onPreview func(stage, step, total int, blob []byte)
		if j.LivePreview {
			onPreview = func(stage, step, total int, blob []byte) {
				j.mu.Lock()
				if len(j.previews) < maxFrames {
					kind := "voxel"
					if len(blob) >= 6 && string(blob[:6]) == "T2MESH" {
						kind = "mesh"
					}
					j.previews = append(j.previews, blob)
					j.Frames = append(j.Frames, frameMeta{
						Stage: stageNames[stage], Step: step, Total: total, Kind: kind})
					j.PreviewSeq = len(j.previews)
				}
				j.PreviewStage = stageNames[stage]
				j.PreviewStep = step
				j.PreviewTotal = total
				j.mu.Unlock()
			}
		}

		mesh, err := s.eng.Generate(j.image, j.pipeline, j.Background, j.Seed, j.Steps, j.Guidance, j.TextureSteps,
			func() {
				setStage("loading models", 0, 0)
			},
			func(stage, step, total int) {
				// The cascade reports its settled resolution as a second
				// upsample event — (achieved, requested), not a step counter.
				// Say so in the log rather than letting it read as "1536/1536".
				if stage == stageUpsample && total > 0 {
					if step < total {
						log.Printf("    cascade reduced to %d³ (%d³ would exceed the token budget)", step, total)
					} else {
						log.Printf("    cascade resolution %d³", step)
					}
				}
				setStage(stageNames[stage], step, total)
			},
			onPreview)

		j.mu.Lock()
		if err != nil {
			j.image = nil
			j.source = nil
			j.State = "error"
			j.Error = err.Error()
		} else {
			j.mesh = mesh
			j.Resolution = mesh.GridRes
		}
		j.mu.Unlock()
		if err == nil {
			setStage("saving generation", 0, 0)
		}

		s.mu.Lock()
		s.active = false
		s.mu.Unlock()
		if s.unloadModelsIfIdle(func() {
			setStage("freeing VRAM", 0, 0)
		}) {
			log.Printf("models unloaded while idle")
		}

		if err == nil {
			setStage("saving generation", 0, 0)
			// Freeze inference/load/unload timings before writing the manifest so
			// the diagnostics survive a server restart. Large mesh-file I/O is not
			// part of the inference duration.
			finishTiming()
			if saveErr := s.persistJob(j); saveErr != nil {
				// The result remains usable for this process. A persistence failure is
				// operational, not a reason to discard an otherwise valid generation.
				log.Printf("persist job %s: %v", j.ID, saveErr)
			}
			j.mu.Lock()
			// If persistence failed or is disabled, keep the bytes in memory so
			// source display and regeneration remain available for this process.
			if j.inputPath != "" {
				j.image = nil
			}
			if j.sourcePath != "" {
				j.source = nil
			}
			j.mu.Unlock()
		} else {
			finishTiming()
		}

		// Read after the run, not before it: the models load lazily inside the
		// generation, so this is the first point at which the backend is settled.
		backendUsed, _, _, _ := s.eng.Info()

		j.mu.Lock()
		if err == nil {
			j.State = "done"
			j.Device = backendUsed
		}
		j.Stage = ""
		j.Step, j.Total = 0, 0
		duration := j.DurationMS
		timings := append([]stageTiming(nil), j.StageTimings...)
		preview := j.LivePreview
		j.mu.Unlock()
		log.Printf("─── Generation timings (job %s) ───", j.ID)
		for _, timing := range timings {
			log.Printf("  %-32s %6.2f s", timing.Stage, float64(timing.Milliseconds)/1000)
		}
		log.Printf("  %-32s %6.2f s", "total", float64(duration)/1000)
		log.Printf("job %s finished in %.1fs (live preview: %t)", j.ID, float64(duration)/1000, preview)
	}
}

// unloadModelsIfIdle rechecks idleness while holding lifecycle, so a generation
// cannot be accepted in the gap between the check and freeing the pipeline.
func (s *server) unloadModelsIfIdle(before func()) bool {
	s.lifecycle.Lock()
	defer s.lifecycle.Unlock()
	s.mu.Lock()
	idle := s.unloadIdle && !s.active && s.queued == 0
	s.mu.Unlock()
	if !idle {
		return false
	}
	if before != nil {
		before()
	}
	return s.eng.Unload()
}

// Switch the compute backend without restarting.
//
// The library picks its backend inside init_best_backend() at model-load time,
// from TRELLIS2_DEVICE, and re-reads it on every load rather than caching it.
// So a switch is: put the name where the DLL's getenv() will see it, then drop
// the loaded models. The next generation comes up on the other backend. No ABI
// change, and no second build tree — which is the point.
//
// Only useful for a build that carries more than one GPU backend;
// cmake-ninja-win64-rocm.bat builds HIP and Vulkan together. A name matching no
// device makes the library fall back to the CPU and say so on stderr, rather
// than quietly using the other one.
//
// Refused while a generation is running: the models in use are the ones that
// would have to be freed.
func (s *server) setDevice(name string) error {
	s.lifecycle.Lock()
	defer s.lifecycle.Unlock()
	s.mu.Lock()
	busy := s.active || s.queued > 0
	s.mu.Unlock()
	if busy {
		return fmt.Errorf("a generation is running")
	}
	if err := os.Setenv("TRELLIS2_DEVICE", name); err != nil {
		return err
	}
	// os.Setenv alone does not reach the library's getenv() on Windows.
	if err := setNativeEnv("TRELLIS2_DEVICE", name); err != nil {
		return err
	}
	s.eng.Unload()
	return nil
}

// What the next model load will use. Empty means the library's own default:
// the first GPU device it finds, else the CPU.
func currentDevice() string {
	if v := os.Getenv("TRELLIS2_DEVICE"); v != "" {
		return v
	}
	return "auto"
}

func (s *server) setUnloadIdle(enabled bool) bool {
	s.lifecycle.Lock()
	defer s.lifecycle.Unlock()
	s.mu.Lock()
	s.unloadIdle = enabled
	idle := !s.active && s.queued == 0
	s.mu.Unlock()
	return enabled && idle && s.eng.Unload()
}

func (s *server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxUpload)
	if err := r.ParseMultipartForm(maxUpload); err != nil {
		http.Error(w, "bad multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer r.MultipartForm.RemoveAll()
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "missing image field", http.StatusBadRequest)
		return
	}
	defer file.Close()
	img, err := readImage(file)
	if err != nil {
		http.Error(w, "invalid image: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(img) == 0 {
		http.Error(w, "empty image", http.StatusBadRequest)
		return
	}

	// Browsers send the untouched selected File separately from the PNG they
	// prepared for inference. API clients that only send `image` still retain
	// that image as their original source.
	source := img
	if sourceFile, _, sourceErr := r.FormFile("source"); sourceErr == nil {
		defer sourceFile.Close()
		source, err = readImage(sourceFile)
		if err != nil || len(source) == 0 {
			if err == nil {
				err = fmt.Errorf("empty image")
			}
			http.Error(w, "invalid source image: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else if sourceErr != http.ErrMissingFile {
		http.Error(w, "invalid source image: "+sourceErr.Error(), http.StatusBadRequest)
		return
	}

	// quality: "coarse" | "512" | "1024" | "1536" (default auto → best available,
	// which stops at 1024: 1536 is an explicit choice, not a default)
	pt := pipelineForQuality(r.FormValue("quality"))

	// Browser uploads are already cleaned so they pass "keep" to prevent a
	// second heuristic pass. Other API clients get automatic cleanup by default.
	background := backgroundAuto
	switch r.FormValue("background") {
	case "keep":
		background = backgroundKeep
	case "black":
		background = backgroundBlack
	case "white":
		background = backgroundWhite
	}

	j := &job{
		ID:           fmt.Sprintf("%016x", rand.Uint64()),
		State:        "queued",
		CreatedAt:    time.Now().UnixMilli(),
		Quality:      r.FormValue("quality"),
		Thumbnail:    r.FormValue("thumbnail"),
		image:        img,
		source:       source,
		pipeline:     pt,
		Background:   background,
		Seed:         formUint(r, "seed", rand.Uint64()%1_000_000),
		Steps:        int(formUint(r, "steps", 12)),
		TextureSteps: int(formUint(r, "texture_steps", 12)),
		Guidance:     formFloat(r, "guidance", 7.5),
		LivePreview:  r.FormValue("preview") == "1", // expensive preview decodes are explicitly opt-in
		keyframes:    int(formUint(r, "keyframes", 0)),
	}
	s.normaliseJob(j)
	s.enqueueJob(w, j)
}

func readImage(r io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, maxImage+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxImage {
		return nil, fmt.Errorf("image exceeds %d MiB", maxImage>>20)
	}
	return data, nil
}

func pipelineForQuality(quality string) int {
	switch quality {
	case "coarse":
		return pipeCoarse
	case "512":
		return pipe512
	case "1024":
		return pipe1024
	case "1536":
		return pipe1536
	default:
		return pipeAuto
	}
}

func (s *server) normaliseJob(j *job) {
	if len(j.Thumbnail) > 256<<10 ||
		(j.Thumbnail != "" && !strings.HasPrefix(j.Thumbnail, "data:image/jpeg;base64,")) {
		j.Thumbnail = ""
	}
	if j.Steps < 1 || j.Steps > 50 {
		j.Steps = 12
	}
	if j.TextureSteps < 1 || j.TextureSteps > 50 {
		j.TextureSteps = 12
	}
	if j.Guidance < 0 || j.Guidance > 20 {
		j.Guidance = 7.5
	}
	if j.keyframes > 8 {
		j.keyframes = 8
	}
	if j.keyframes < 0 || !j.LivePreview {
		j.keyframes = 0
	}
}

func (s *server) enqueueJob(w http.ResponseWriter, j *job) bool {
	s.lifecycle.Lock()
	s.mu.Lock()
	select {
	case s.q <- j:
		s.jobs[j.ID] = j
		s.queued++
		s.mu.Unlock()
		s.lifecycle.Unlock()
		writeJSON(w, map[string]string{"job": j.ID})
		return true
	default:
		s.mu.Unlock()
		s.lifecycle.Unlock()
		http.Error(w, "queue full, try again later", http.StatusServiceUnavailable)
		return false
	}
}

// handleRegenerate creates a new job from the immutable, processed input saved
// with a completed generation. No image bytes need to cross the network again.
func (s *server) handleRegenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	old := s.getJob(r, "/api/regenerate/")
	if old == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}

	old.mu.Lock()
	state := old.State
	input := append([]byte(nil), old.image...)
	source := append([]byte(nil), old.source...)
	inputPath, sourcePath := old.inputPath, old.sourcePath
	quality, thumbnail := old.Quality, old.Thumbnail
	seed, steps := old.Seed, old.Steps
	textureSteps, guidance := old.TextureSteps, old.Guidance
	old.mu.Unlock()
	if state != "done" {
		http.Error(w, "generation is not complete", http.StatusConflict)
		return
	}
	// source.img was the processed input in the first persistence format. That
	// makes legacy saved jobs regeneratable even though their exact original can
	// no longer be recovered retroactively.
	if inputPath == "" {
		inputPath = sourcePath
	}
	var err error
	if len(input) == 0 && inputPath != "" {
		input, err = os.ReadFile(inputPath)
	}
	if err != nil || len(input) == 0 {
		http.Error(w, "saved generation input is unavailable", http.StatusConflict)
		return
	}
	if len(source) == 0 && sourcePath != "" {
		source, err = os.ReadFile(sourcePath)
		if err != nil {
			source = nil
		}
	}
	if len(source) == 0 {
		source = append([]byte(nil), input...)
	}

	if requested := r.FormValue("quality"); requested != "" {
		quality = requested
	}
	j := &job{
		ID: fmt.Sprintf("%016x", rand.Uint64()), State: "queued",
		CreatedAt: time.Now().UnixMilli(), Quality: quality, Thumbnail: thumbnail,
		image: input, source: source, pipeline: pipelineForQuality(quality),
		// The persisted input is already background-processed, so a second
		// heuristic pass over it would be wrong regardless of what the original
		// request asked for.
		Background: backgroundKeep,
		Seed:       formUint(r, "seed", seed), Steps: int(formUint(r, "steps", uint64(steps))),
		TextureSteps: int(formUint(r, "texture_steps", uint64(textureSteps))),
		Guidance:     formFloat(r, "guidance", guidance),
		LivePreview:  r.FormValue("preview") == "1",
		keyframes:    int(formUint(r, "keyframes", 0)),
	}
	s.normaliseJob(j)
	s.enqueueJob(w, j)
}

func (s *server) getJob(r *http.Request, prefix string) *job {
	id := r.URL.Path[len(prefix):]
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.jobs[id]
}

func (s *server) handleJob(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/job/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	if r.Method == http.MethodDelete {
		j.mu.Lock()
		state, dir := j.State, j.persistDir
		j.mu.Unlock()
		if state != "done" && state != "error" {
			http.Error(w, "generation is still active", http.StatusConflict)
			return
		}
		if dir != "" {
			if err := os.RemoveAll(dir); err != nil {
				http.Error(w, "delete persisted generation: "+err.Error(), http.StatusInternalServerError)
				return
			}
		}
		s.mu.Lock()
		delete(s.jobs, j.ID)
		s.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "GET or DELETE only", http.StatusMethodNotAllowed)
		return
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	writeJSON(w, j)
}

// handleSource serves the original upload, retained byte-for-byte for the
// full-screen showcase and other future use outside the inference pipeline.
func (s *server) handleSource(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/source/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	j.mu.Lock()
	image := j.source
	if len(image) == 0 {
		image = j.image // compatibility for an in-memory job from an older server
	}
	path := j.sourcePath
	j.mu.Unlock()
	if len(image) > 0 {
		w.Header().Set("Content-Type", http.DetectContentType(image))
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		w.Write(image)
		return
	}
	if path == "" {
		http.Error(w, "source image was not retained for this generation", http.StatusNotFound)
		return
	}
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	http.ServeFile(w, r, path)
}

type jobSummary struct {
	ID            string `json:"id"`
	CreatedAt     int64  `json:"createdAt"`
	Quality       string `json:"quality,omitempty"`
	Device        string `json:"device,omitempty"`
	Thumbnail     string `json:"thumbnail,omitempty"`
	PreviewSeq    int    `json:"previewSeq"`
	Source        bool   `json:"sourceAvailable"`
	Regeneratable bool   `json:"regeneratable"`
}

func (s *server) handleJobs(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	jobs := make([]*job, 0, len(s.jobs))
	for _, j := range s.jobs {
		jobs = append(jobs, j)
	}
	s.mu.Unlock()

	if r.Method == http.MethodDelete {
		deleted := 0
		for _, j := range jobs {
			j.mu.Lock()
			state, dir := j.State, j.persistDir
			j.mu.Unlock()
			if state != "done" && state != "error" {
				continue
			}
			if dir != "" {
				if err := os.RemoveAll(dir); err != nil {
					http.Error(w, "delete persisted generation: "+err.Error(), http.StatusInternalServerError)
					return
				}
			}
			s.mu.Lock()
			if s.jobs[j.ID] == j {
				delete(s.jobs, j.ID)
				deleted++
			}
			s.mu.Unlock()
		}
		writeJSON(w, map[string]int{"deleted": deleted})
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "GET or DELETE only", http.StatusMethodNotAllowed)
		return
	}

	out := make([]jobSummary, 0, len(jobs))
	for _, j := range jobs {
		j.mu.Lock()
		if j.State == "done" {
			out = append(out, jobSummary{
				ID: j.ID, CreatedAt: j.CreatedAt, Quality: j.Quality, Device: j.Device,
				Thumbnail: j.Thumbnail, PreviewSeq: j.PreviewSeq,
				Source:        len(j.source) > 0 || j.sourcePath != "",
				Regeneratable: len(j.image) > 0 || j.inputPath != "" || j.sourcePath != "",
			})
		}
		j.mu.Unlock()
	}
	sort.Slice(out, func(i, k int) bool { return out[i].CreatedAt > out[k].CreatedAt })
	writeJSON(w, out)
}

func (s *server) handleMesh(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/mesh/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	mesh, err := s.loadJobMesh(j)
	if err != nil {
		http.Error(w, "mesh not ready: "+err.Error(), http.StatusConflict)
		return
	}
	writeMesh(w, mesh)
}

// writeMesh emits T2MESH01 geometry or T2MESH03 geometry followed by six-float
// PBR attributes. It is shared by generated and component-cleanup preview meshes.
func writeMesh(w http.ResponseWriter, mesh *meshData) {
	w.Header().Set("Content-Type", "application/octet-stream")
	if err := writeMeshBinary(w, mesh); err != nil {
		log.Printf("write mesh response: %v", err)
	}
}

func writeMeshBinary(w io.Writer, mesh *meshData) error {
	textured := len(mesh.PBR) == 6*mesh.NVerts
	if textured {
		if _, err := w.Write([]byte("T2MESH03")); err != nil {
			return err
		}
	} else {
		if _, err := w.Write([]byte("T2MESH01")); err != nil {
			return err
		}
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(mesh.NVerts)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(mesh.NTris)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, mesh.Verts); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, mesh.Normals); err != nil {
		return err
	}
	if textured {
		if err := binary.Write(w, binary.LittleEndian, mesh.PBR); err != nil {
			return err
		}
	}
	return binary.Write(w, binary.LittleEndian, mesh.Tris)
}

type exportOptions struct {
	textureSize     int
	componentFilter int
	printWrap       bool
	alphaRatio      float32
	offsetRatio     float32
	quadRemesh      bool
	targetQuads     int
	quadAdaptivity  float32
	normalMap       bool
}

func parseExportOptions(r *http.Request) exportOptions {
	o := exportOptions{
		textureSize:     2048,
		componentFilter: 2,          // safe default: preserve every connected component
		alphaRatio:      0.005,      // detail size: 0.5% of the bbox diagonal
		offsetRatio:     0.005 / 30, // shell standoff ~alpha/30 (CGAL guideline)
		targetQuads:     20000,      // density hint for the quad remesher, not a face count
		quadAdaptivity:  1,          // curvature-adaptive density
		normalMap:       true,       // transfer the dense mesh's detail, not just its material
	}
	if n := int(formUint(r, "tex", uint64(o.textureSize))); n >= 256 && n <= 4096 {
		o.textureSize = n
	}
	switch r.FormValue("components") {
	case "tiny":
		o.componentFilter = 0
	case "largest":
		o.componentFilter = 1
	}
	o.printWrap = r.FormValue("print") == "1" || r.FormValue("print") == "true"
	o.quadRemesh = r.FormValue("quad") == "1" || r.FormValue("quad") == "true"
	// Default-on, so an absent parameter keeps the detail transfer. Only an
	// explicit "0"/"false" switches it off.
	if v := r.FormValue("normalmap"); v != "" {
		o.normalMap = v != "0" && v != "false"
	}
	if n := int(formFloat(r, "quads", float32(o.targetQuads))); n >= 100 && n <= 500000 {
		o.targetQuads = n
	}
	if a := formFloat(r, "adaptivity", o.quadAdaptivity); a >= 0 && a <= 1 {
		o.quadAdaptivity = a
	}
	if pct := formFloat(r, "alpha", o.alphaRatio*100); pct >= 0.01 && pct <= 50 {
		o.alphaRatio = pct / 100
	}
	if pct := formFloat(r, "offset", o.offsetRatio*100); pct >= 0.001 && pct <= 50 {
		o.offsetRatio = pct / 100
	}
	return o
}

func (o exportOptions) prepareKey() string {
	key := strconv.Itoa(o.componentFilter)
	if o.printWrap {
		key = fmt.Sprintf("%s-wrap-%g-%g", key, o.alphaRatio, o.offsetRatio)
	}
	if o.quadRemesh {
		key = fmt.Sprintf("%s-quad-%d-%g", key, o.targetQuads, o.quadAdaptivity)
	}
	return key
}

// The normal map changes the GLB but not the prepared geometry, so it belongs
// here and not in prepareKey. It is only reachable on the projected path, so
// toggling it in dense mode must not invalidate a cached bake.
func (o exportOptions) glbKey() string {
	key := fmt.Sprintf("%d-%s", o.textureSize, o.prepareKey())
	if (o.printWrap || o.quadRemesh) && !o.normalMap {
		key += "-nonormalmap"
	}
	return key
}

func (s *server) preparedExportMesh(j *job, o exportOptions, sl *stageLog) (*meshData, error) {
	// Keep-all preview is the saved source object itself: no copying, normal
	// recomputation, topology cleanup, or other opportunity to alter it.
	if o.componentFilter == 2 && !o.printWrap && !o.quadRemesh {
		return s.loadJobMesh(j)
	}
	key := o.prepareKey()
	j.mu.Lock()
	if j.exportKey == key && j.exportMesh != nil {
		mesh := j.exportMesh
		j.mu.Unlock()
		return mesh, nil
	}
	j.mu.Unlock()
	var source *meshData
	err := sl.track("load source mesh", func() (e error) {
		source, e = s.loadJobMesh(j)
		return
	})
	if err != nil {
		return nil, err
	}
	var mesh *meshData
	if o.printWrap {
		log.Printf("  alpha wrap (CGAL, detail %.3f%%) on %d tris ...", 100*o.alphaRatio, source.NTris)
		err = sl.trackLive("alpha wrap (CGAL)", "wrapping", func() (e error) {
			mesh, e = s.eng.PreparePrintMesh(source, o.componentFilter, o.alphaRatio, o.offsetRatio)
			return
		})
	} else {
		err = sl.track("component filter", func() (e error) {
			mesh, e = s.eng.PrepareMesh(source, o.componentFilter)
			return
		})
	}
	if err != nil {
		return nil, err
	}
	// Quad remeshing runs after the optional wrap: Alpha Wrap hands it a clean
	// 2-manifold, which is the input it wants. The combination is still not
	// guaranteed watertight - the quad stage can reintroduce boundaries - so the
	// stats below are what the UI must show, not a printable claim.
	if o.quadRemesh {
		var quadded *meshData
		var stats *QuadStats
		log.Printf("  quad remesh (target %d quads, adaptivity %.2f) on %d tris ...",
			o.targetQuads, o.quadAdaptivity, mesh.NTris)
		qerr := sl.trackLive("quad remesh (AutoRemesher)", "remeshing", func() (e error) {
			quadded, stats, e = s.eng.PrepareQuadMesh(mesh, 2, o.targetQuads, o.quadAdaptivity)
			return
		})
		if qerr != nil {
			return nil, qerr
		}
		mesh = quadded
		j.mu.Lock()
		j.quadStats = stats
		j.mu.Unlock()
		if stats != nil {
			log.Printf("    %d quads / %d tris / %d n-gons, %d boundary edges, %.1f%% area kept",
				stats.Quads, stats.Triangles, stats.NGons, stats.BoundaryEdges, 100*stats.AreaRetained)
		}
	}
	j.mu.Lock()
	j.exportMesh, j.exportKey = mesh, key
	// A restored source can always be re-read. Avoid pinning duplicate full-size
	// source and component-cleanup meshes in server memory.
	if j.persistDir != "" {
		j.mesh = nil
	}
	j.mu.Unlock()
	return mesh, nil
}

func (s *server) handleExportPreview(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/export-preview/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	o := parseExportOptions(r)
	sl := newStageLog()
	// Serialize per job, like the GLB download already does. Without this a
	// browser retry (or two viewers) starts a second full preparation while the
	// first is still running - with quad remeshing that means two multi-minute
	// runs of several GB competing for the same cores, which makes both slower
	// and looks like a hang. Waiting here means the retry hits the cache.
	j.exportMu.Lock()
	mesh, err := s.preparedExportMesh(j, o, sl)
	j.exportMu.Unlock()
	if err != nil {
		http.Error(w, "prepare export: "+err.Error(), http.StatusInternalServerError)
		return
	}
	sl.dump("Export preview timings (job " + j.ID + ")")
	// Quad quality travels as headers because the body is the binary mesh. The
	// viewer shows these rather than implying a guarantee the stage cannot give:
	// boundary edges > 0 means the surface is open.
	if o.quadRemesh {
		j.mu.Lock()
		qs := j.quadStats
		j.mu.Unlock()
		if qs != nil {
			w.Header().Set("X-Quad-Quads", strconv.Itoa(qs.Quads))
			w.Header().Set("X-Quad-Triangles", strconv.Itoa(qs.Triangles))
			w.Header().Set("X-Quad-Ngons", strconv.Itoa(qs.NGons))
			w.Header().Set("X-Quad-Boundary-Edges", strconv.Itoa(qs.BoundaryEdges))
			w.Header().Set("X-Quad-Area-Retained", strconv.FormatFloat(float64(qs.AreaRetained), 'f', 4, 32))
		}
	}
	writeMesh(w, mesh)
}

// handlePreview streams a recorded intermediate-preview frame for a job. With
// ?seq=N it returns that frame by index (the browser fetches every frame it is
// missing so the scrubber can replay the whole generation); without it, the
// latest frame (back-compat live view). Frames are T2VOX01 voxel sets or
// T2MESH0* keyframe meshes; the client dispatches on the magic.
func (s *server) handlePreview(w http.ResponseWriter, r *http.Request) {
	j := s.getJob(r, "/api/preview/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	j.mu.Lock()
	seq := -1
	if q := r.URL.Query().Get("seq"); q != "" {
		if i, err := strconv.Atoi(q); err == nil {
			seq = i
		}
	} else if j.PreviewSeq > 0 {
		seq = j.PreviewSeq - 1
	}
	j.mu.Unlock()
	blob, err := loadJobPreview(j, seq)
	if err != nil {
		http.Error(w, "no such preview frame", http.StatusConflict)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(blob)
}

// handleGLB writes the exact prepared preview mesh into a GLB. Print wraps with
// source PBR take the upstream-style UV-atlas reprojection path; ordinary dense
// meshes retain their compact vertex material. Both stages are settings-cached.
func (s *server) handleGLB(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "GET or HEAD only", http.StatusMethodNotAllowed)
		return
	}
	j := s.getJob(r, "/api/glb/")
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	o := parseExportOptions(r)
	key := o.glbKey()

	j.exportMu.Lock()
	sl := newStageLog()
	defer sl.dump("Export timings (job " + j.ID + ")")
	glb, err := func() ([]byte, error) {
		mesh, err := s.preparedExportMesh(j, o, sl)
		if err != nil {
			return nil, fmt.Errorf("prepare export: %w", err)
		}
		j.mu.Lock()
		glb, cached := j.glb, j.glbKey == key && j.glb != nil
		j.mu.Unlock()
		if cached {
			return glb, nil
		}
		started := time.Now()
		// Kept next to the cached GLB rather than written straight to the job:
		// the cache key encodes whether a normal map was baked, so the stats and
		// the bytes they describe have to be published together or a later bake
		// under a different key would leave stale numbers behind.
		var bakedNormalMap *NormalMapStats
		// Either replacement path produces new vertices, so the source material
		// has to be reprojected onto them. This used to check printWrap only,
		// which silently gave a quad export vertex colours instead of the atlas.
		if o.printWrap || o.quadRemesh {
			var source *meshData
			if sourceErr := sl.track("load projection source", func() (e error) {
				source, e = s.loadJobMesh(j)
				return
			}); sourceErr != nil {
				return nil, fmt.Errorf("load projection source: %w", sourceErr)
			}
			if len(source.PBR) == 6*source.NVerts {
				detail := ""
				if o.normalMap {
					detail = " + tangent-space normal map"
				}
				log.Printf("  projected GLB bake (%d px atlas%s) on %d tris ...", o.textureSize, detail, mesh.NTris)
				err = sl.trackLive("projected GLB bake (unwrap + per-texel PBR"+detail+")", "baking", func() (e error) {
					glb, e = s.eng.BakeProjectedGLB(mesh, source, o.textureSize, o.componentFilter, o.normalMap)
					return
				})
				if err == nil && o.normalMap {
					ns := s.eng.LastNormalMapStats()
					bakedNormalMap = &ns
					if ns.CoveredTexels > 0 {
						log.Printf("  normal map: %d of %d covered texels left flat (%.1f%%)",
							ns.RejectedTexels, ns.CoveredTexels,
							100*float64(ns.RejectedTexels)/float64(ns.CoveredTexels))
					}
				}
			} else {
				err = sl.track("GLB bake (vertex colours)", func() (e error) {
					glb, e = s.eng.BakeGLB(mesh, o.textureSize, 2 /*already prepared*/)
					return
				})
			}
		} else {
			err = sl.track("GLB bake (vertex colours)", func() (e error) {
				glb, e = s.eng.BakeGLB(mesh, o.textureSize, 2 /*already prepared*/)
				return
			})
		}
		if err != nil {
			return nil, fmt.Errorf("bake glb: %w", err)
		}
		// The bake is the largest single export cost on dense meshes and the UV
		// unwrap usually dominates it, so report the split instead of one number.
		bt := s.eng.LastBakeTimings()
		sl.add("- xatlas UV unwrap", time.Duration(float64(bt.Unwrap)*float64(time.Second)))
		sl.add("- rasterize coverage", time.Duration(float64(bt.Rasterize)*float64(time.Second)))
		if bt.Projection > 0 {
			sl.add("- closest-surface projection ("+s.eng.ProjectionBackend()+")",
				time.Duration(float64(bt.Projection)*float64(time.Second)))
		}
		sl.add("- texel fill", time.Duration(float64(bt.TexelFill)*float64(time.Second)))
		sl.add("- inpaint + PNG + glTF", time.Duration(float64(bt.Encode)*float64(time.Second)))
		j.mu.Lock()
		j.glb, j.glbKey, j.normalMapStats = glb, key, bakedNormalMap
		j.mu.Unlock()
		log.Printf("job %s baked %.1f MiB GLB in %.1fs (%s)", j.ID,
			float64(len(glb))/(1<<20), time.Since(started).Seconds(), key)
		return glb, nil
	}()
	j.exportMu.Unlock()
	if err != nil {
		w.Header().Set("X-Trellis-Error", err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	j.mu.Lock()
	finishedAt := j.FinishedAt
	j.mu.Unlock()
	var modTime time.Time
	if finishedAt > 0 {
		modTime = time.UnixMilli(finishedAt)
	}
	w.Header().Set("Content-Type", "model/gltf-binary")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"trellis2-%s.glb\"", j.ID))
	w.Header().Set("ETag", fmt.Sprintf("\"%s-%s\"", j.ID, key))
	// Normal map quality travels as headers, like the quad statistics: the body
	// is the binary asset, and the viewer cannot show the map itself, so these
	// numbers are the only feedback on how much detail actually transferred.
	j.mu.Lock()
	nms := j.normalMapStats
	qs := j.quadStats
	j.mu.Unlock()
	if nms != nil {
		w.Header().Set("X-Normal-Map", "1")
		w.Header().Set("X-Normal-Map-Texels", strconv.Itoa(int(nms.CoveredTexels)))
		w.Header().Set("X-Normal-Map-Rejected", strconv.Itoa(int(nms.RejectedTexels)))
	}
	// The viewer previews this exact GLB for either replacement path, so the
	// quad quality figures have to be reachable here too and not only from
	// /api/export-preview - otherwise previewing a quad remesh through the baked
	// GLB would silently drop the boundary-edge and retained-area readout.
	if o.quadRemesh && qs != nil {
		w.Header().Set("X-Quad-Quads", strconv.Itoa(qs.Quads))
		w.Header().Set("X-Quad-Triangles", strconv.Itoa(qs.Triangles))
		w.Header().Set("X-Quad-Ngons", strconv.Itoa(qs.NGons))
		w.Header().Set("X-Quad-Boundary-Edges", strconv.Itoa(qs.BoundaryEdges))
		w.Header().Set("X-Quad-Area-Retained", strconv.FormatFloat(float64(qs.AreaRetained), 'f', 4, 32))
	}
	// ServeContent supplies Content-Length plus byte-range support. Browsers can
	// stream these 100+ MiB assets directly to disk and resume an interrupted
	// transfer instead of JavaScript copying the whole response into a Blob.
	http.ServeContent(w, r, fmt.Sprintf("trellis2-%s.glb", j.ID),
		modTime, bytes.NewReader(glb))
}

func (s *server) info() map[string]interface{} {
	backend, caps, textured, loaded := s.eng.Info()
	qualities := []string{"coarse"}
	if caps&cap512 != 0 {
		qualities = append(qualities, "512")
	}
	if caps&cap1024 != 0 {
		qualities = append(qualities, "1024")
	}
	if caps&cap1536 != 0 {
		qualities = append(qualities, "1536")
	}
	// "best" is what auto picks, and auto stops at 1024 — 1536 needs a card that
	// can hold its decode, so offering it is right but defaulting to it is not.
	best := "coarse"
	if caps&cap1024 != 0 {
		best = "1024"
	} else if caps&cap512 != 0 {
		best = "512"
	}
	s.mu.Lock()
	unloadIdle := s.unloadIdle
	generationActive := s.active || s.queued > 0
	s.mu.Unlock()
	return map[string]interface{}{
		"backend":            backend,
		"qualities":          qualities,
		"best":               best,
		"textured":           textured,
		"models_loaded":      loaded,
		"unload_idle":        unloadIdle,
		"device":             currentDevice(),
		"generation_active":  generationActive,
		"print_remesh":       s.eng.HasPrintRemesh(),
		"quad_remesh":        s.eng.HasQuadRemesh(),
		"projection_backend": s.eng.ProjectionBackend(),
		"defaults": map[string]interface{}{
			"steps":         12,
			"guidance":      7.5,
			"texture_steps": 12,
		},
	}
}

func (s *server) handleInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.info())
}

func (s *server) handleSettings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form: "+err.Error(), http.StatusBadRequest)
		return
	}
	if d := r.FormValue("device"); d != "" {
		if err := s.setDevice(d); err != nil {
			http.Error(w, "device: "+err.Error(), http.StatusConflict)
			return
		}
		log.Printf("compute device set to %q; models unloaded, the next generation reloads on it", d)
	}
	if v := r.FormValue("unload_idle"); v != "" {
		enabled := v == "1" || v == "true" || v == "on"
		if s.setUnloadIdle(enabled) {
			log.Printf("models unloaded while idle")
		}
	}
	writeJSON(w, s.info())
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func formUint(r *http.Request, key string, def uint64) uint64 {
	if v := r.FormValue(key); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}

func formFloat(r *http.Request, key string, def float32) float32 {
	if v := r.FormValue(key); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			return float32(f)
		}
	}
	return def
}

func main() {
	libDefault := "../build/libtrellis2.so"
	if runtime.GOOS == "windows" {
		libDefault = "../build/bin/Release/trellis2.dll"
	}
	libPath := flag.String("lib", libDefault, "path to libtrellis2 shared library (.so / .dll)")
	ggufDir := flag.String("ggufs", "../ggufs", "directory with the model ggufs")
	dino := flag.String("dino", "", "dino gguf (default <ggufs>/dino_f16.gguf)")
	flow := flag.String("flow", "", "ss_flow gguf (default <ggufs>/ss_flow_f16.gguf)")
	dec := flag.String("dec", "", "ss_dec gguf (default <ggufs>/ss_dec_f16.gguf)")
	slat := flag.String("slat", "", "512 shape-slat flow gguf (default <ggufs>/slat_flow_f16.gguf)")
	slatHR := flag.String("slat-hr", "", "1024 shape-slat flow gguf (default <ggufs>/slat_flow_1024_f16.gguf)")
	shapeDec := flag.String("shape-dec", "", "shape decoder gguf (default <ggufs>/shape_dec_f16.gguf)")
	shapeEnc := flag.String("shape-enc", "", "shape encoder gguf (default <ggufs>/shape_enc_f16.gguf)")
	texDec := flag.String("tex-dec", "", "texture decoder gguf (default <ggufs>/tex_dec_f16.gguf)")
	texSlat := flag.String("tex-slat", "", "512 texture-slat flow gguf (default <ggufs>/tex_slat_flow_512_f16.gguf)")
	texSlatHR := flag.String("tex-slat-hr", "", "1024 texture-slat flow gguf (default <ggufs>/tex_slat_flow_1024_f16.gguf)")
	coarse := flag.Bool("coarse", false, "coarse marching-cubes path only (skip shape-SLAT models)")
	no1024 := flag.Bool("no-1024", false, "disable the 1024 cascade (512 fine max)")
	noTexture := flag.Bool("no-texture", false, "disable PBR texturing (geometry only)")
	addr := flag.String("addr", ":8742", "listen address")
	storeDir := flag.String("store", "../generations", "durable completed-generation directory (empty disables persistence)")
	unloadIdle := flag.Bool("unload-idle", false, "start with models unloaded and release them after each idle generation")
	flag.Parse()

	// Recorded previews: capture one voxel frame per SS step (finer than the
	// library's ~4-frame default) so the scrubber has a smooth structure-forming
	// sequence to replay. The library reads T2_PREVIEW_STRIDE; respect an
	// explicit override, else default to per-step.
	if os.Getenv("T2_PREVIEW_STRIDE") == "" {
		os.Setenv("T2_PREVIEW_STRIDE", "1")
	}

	// ggml's CUDA-graph capture overflows the calling thread's stack and takes the
	// whole process down with STATUS_STACK_OVERFLOW (0xc00000fd). Not
	// workload-dependent: the coarse 64^3 path dies as readily as the 1024^3
	// cascade.
	//
	// This was originally read as the small Go-managed thread the purego bridge
	// calls in on, because a native executable seemed fine. That no longer holds
	// after the ggml 0.19 update: examples/ss_sample.exe crashes the same way,
	// intermittently — 2 of 5 runs measured on 2026-08-14. So it is the graph
	// path's stack usage, not Go's thread.
	//
	// The library now disables graphs itself in init_best_backend(), which covers
	// every consumer rather than only this server; this block is kept as a
	// belt-and-braces for an older libtrellis2 and is a no-op against a current
	// one. os.Setenv alone does not reach the library's getenv() on Windows — see
	// setNativeEnv. To put graphs back for investigation, set TRELLIS2_CUDA_GRAPHS=1;
	// GGML_CUDA_DISABLE_GRAPHS=0 will not do it, as ggml only tests for presence.
	if os.Getenv("GGML_CUDA_DISABLE_GRAPHS") == "" {
		os.Setenv("GGML_CUDA_DISABLE_GRAPHS", "1")
		if err := setNativeEnv("GGML_CUDA_DISABLE_GRAPHS", "1"); err != nil {
			log.Printf("warning: could not disable CUDA graphs (%v); a GPU generation may crash", err)
		}
	}

	pick := func(explicit, name string) string {
		if explicit != "" {
			return explicit
		}
		return filepath.Join(*ggufDir, name)
	}
	// The fine (dual-grid) path needs the two shape-SLAT models; the 1024 cascade
	// additionally needs the 1024 model. Missing files degrade gracefully.
	slatPath, shapePath, slatHRPath := "", "", ""
	shapeEncPath, texDecPath, texSlatPath, texSlatHRPath := "", "", "", ""
	if !*coarse {
		slatPath = pick(*slat, "slat_flow_f16.gguf")
		shapePath = pick(*shapeDec, "shape_dec_f16.gguf")
		if !fileExists(slatPath) || !fileExists(shapePath) {
			log.Printf("shape-SLAT models not found, using coarse path")
			slatPath, shapePath = "", ""
		} else {
			if !*no1024 {
				slatHRPath = pick(*slatHR, "slat_flow_1024_f16.gguf")
				if !fileExists(slatHRPath) {
					log.Printf("1024 model not found, 512 fine max")
					slatHRPath = ""
				}
			}
			// The validated PBR path re-encodes the decoded dual grid, so all three
			// model families are required: shape encoder, texture decoder, and flow.
			if !*noTexture {
				shapeEncPath = pick(*shapeEnc, "shape_enc_f16.gguf")
				texDecPath = pick(*texDec, "tex_dec_f16.gguf")
				texSlatPath = pick(*texSlat, "tex_slat_flow_512_f16.gguf")
				if fileExists(shapeEncPath) && fileExists(texDecPath) && fileExists(texSlatPath) {
					texSlatHRPath = pick(*texSlatHR, "tex_slat_flow_1024_f16.gguf")
					if !fileExists(texSlatHRPath) {
						if slatHRPath != "" {
							log.Printf("1024 texture model not found, using complete 512 textured pipeline")
							slatHRPath = ""
						}
						texSlatHRPath = ""
					}
				} else {
					log.Printf("shape encoder or texture models not found, geometry only")
					shapeEncPath, texDecPath, texSlatPath = "", "", ""
				}
			}
		}
	}

	eng, err := newEngine(*libPath, pick(*dino, "dino_f16.gguf"),
		pick(*flow, "ss_flow_f16.gguf"), pick(*dec, "ss_dec_f16.gguf"),
		slatPath, slatHRPath, shapePath,
		shapeEncPath, texDecPath, texSlatPath, texSlatHRPath, *unloadIdle)
	if err != nil {
		log.Fatal(err)
	}
	backend, caps, textured, loaded := eng.Info()
	mode := "coarse (marching cubes)"
	if caps&cap1536 != 0 {
		mode = "1536 cascade (+ 1024 cascade, 512 fine, coarse)"
	} else if caps&cap1024 != 0 {
		mode = "1024 cascade (+ 512 fine, coarse)"
	} else if caps&cap512 != 0 {
		mode = "512 fine (+ coarse)"
	}
	if loaded {
		log.Printf("models loaded, backend: %s, qualities: %s, PBR: %v", backend, mode, textured)
	} else {
		log.Printf("models configured for lazy load, backend: %s, qualities: %s, PBR: %v", backend, mode, textured)
	}

	s := &server{
		eng: eng, jobs: map[string]*job{}, q: make(chan *job, 8),
		storeDir: *storeDir, unloadIdle: *unloadIdle,
	}
	restored, restoreErr := s.restoreJobs()
	if restoreErr != nil {
		log.Printf("job persistence disabled: %v", restoreErr)
	} else if restored > 0 {
		log.Printf("restored %d completed generation(s) from %s", restored, *storeDir)
	}
	go s.worker()

	web, err := fs.Sub(webFS, "web")
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	indexPage, err := fs.ReadFile(web, "index.html")
	if err != nil {
		log.Fatal(err)
	}
	mux.HandleFunc("/showcase", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/showcase" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexPage)
	})
	mux.Handle("/", http.FileServer(http.FS(web)))
	mux.HandleFunc("/api/info", s.handleInfo)
	mux.HandleFunc("/api/settings", s.handleSettings)
	mux.HandleFunc("/api/generate", s.handleGenerate)
	mux.HandleFunc("/api/regenerate/", s.handleRegenerate)
	mux.HandleFunc("/api/jobs", s.handleJobs)
	mux.HandleFunc("/api/job/", s.handleJob)
	mux.HandleFunc("/api/source/", s.handleSource)
	mux.HandleFunc("/api/mesh/", s.handleMesh)
	mux.HandleFunc("/api/export-preview/", s.handleExportPreview)
	mux.HandleFunc("/api/preview/", s.handlePreview)
	mux.HandleFunc("/api/glb/", s.handleGLB)

	log.Printf("listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
