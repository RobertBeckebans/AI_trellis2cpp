package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const persistedJobVersion = 1

// persistedJob is deliberately independent of the public job JSON. Runtime
// state, caches, and locks never reach disk; only the data needed to restore a
// completed generation is retained.
type persistedJob struct {
	Version      int           `json:"version"`
	ID           string        `json:"id"`
	CreatedAt    int64         `json:"createdAt"`
	StartedAt    int64         `json:"startedAt,omitempty"`
	FinishedAt   int64         `json:"finishedAt,omitempty"`
	DurationMS   int64         `json:"durationMs,omitempty"`
	Quality      string        `json:"quality,omitempty"`
	Device       string        `json:"device,omitempty"`
	Thumbnail    string        `json:"thumbnail,omitempty"`
	Pipeline     int           `json:"pipeline"`
	Seed         uint64        `json:"seed"`
	Steps        int           `json:"steps"`
	TextureSteps int           `json:"textureSteps"`
	Guidance     float32       `json:"guidance"`
	// Absent in v1 manifests; those restore as 0 (auto), which is what they ran
	// with before the field existed.
	Background   int           `json:"background,omitempty"`
	Frames       []frameMeta   `json:"frames,omitempty"`
	LivePreview  bool          `json:"livePreview,omitempty"`
	StageTimings []stageTiming `json:"stageTimings,omitempty"`
}

func persistedFrameName(i int) string {
	return filepath.Join("frames", fmt.Sprintf("%06d.bin", i))
}

// persistJob writes a complete job into a temporary sibling directory and then
// renames that directory into place. Startup therefore sees either the previous
// complete job or no job, never a half-written mesh/frame set.
func (s *server) persistJob(j *job) error {
	if s.storeDir == "" {
		return nil
	}

	j.mu.Lock()
	if j.mesh == nil {
		j.mu.Unlock()
		return fmt.Errorf("job %s has no mesh", j.ID)
	}
	manifest := persistedJob{
		Version: persistedJobVersion, ID: j.ID, CreatedAt: j.CreatedAt,
		StartedAt: j.StartedAt, FinishedAt: j.FinishedAt, DurationMS: j.DurationMS,
		Quality: j.Quality, Device: j.Device, Thumbnail: j.Thumbnail,
		Pipeline: j.pipeline, Seed: j.Seed, Steps: j.Steps,
		TextureSteps: j.TextureSteps, Guidance: j.Guidance,
		Background: j.Background,
		Frames: append([]frameMeta(nil), j.Frames...), LivePreview: j.LivePreview,
		StageTimings: append([]stageTiming(nil), j.StageTimings...),
	}
	mesh := j.mesh
	input := j.image
	source := j.source
	if len(source) == 0 {
		source = input
	}
	previews := append([][]byte(nil), j.previews...)
	j.mu.Unlock()

	if len(previews) != len(manifest.Frames) {
		return fmt.Errorf("job %s has %d previews but %d frame records",
			j.ID, len(previews), len(manifest.Frames))
	}
	if err := os.MkdirAll(s.storeDir, 0o755); err != nil {
		return fmt.Errorf("create job store: %w", err)
	}
	tmp, err := os.MkdirTemp(s.storeDir, "."+j.ID+".tmp-")
	if err != nil {
		return fmt.Errorf("create temporary job directory: %w", err)
	}
	defer os.RemoveAll(tmp)

	if err := writeMeshFile(filepath.Join(tmp, "mesh.t2mesh"), mesh); err != nil {
		return err
	}
	if len(source) > 0 {
		if err := os.WriteFile(filepath.Join(tmp, "source.img"), source, 0o644); err != nil {
			return fmt.Errorf("write source image: %w", err)
		}
	}
	if len(input) > 0 {
		if err := os.WriteFile(filepath.Join(tmp, "input.img"), input, 0o644); err != nil {
			return fmt.Errorf("write generation input: %w", err)
		}
	}
	if len(previews) > 0 {
		if err := os.Mkdir(filepath.Join(tmp, "frames"), 0o755); err != nil {
			return fmt.Errorf("create frame directory: %w", err)
		}
		for i, blob := range previews {
			if err := os.WriteFile(filepath.Join(tmp, persistedFrameName(i)), blob, 0o644); err != nil {
				return fmt.Errorf("write preview %d: %w", i, err)
			}
		}
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("encode manifest: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(filepath.Join(tmp, "manifest.json"), data, 0o644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}

	finalDir := filepath.Join(s.storeDir, j.ID)
	if err := os.Rename(tmp, finalDir); err != nil {
		return fmt.Errorf("commit job %s: %w", j.ID, err)
	}

	// Frames are immutable and now durable, so release their duplicate in-memory
	// copies. The preview endpoint transparently reads them from finalDir.
	j.mu.Lock()
	j.persistDir = finalDir
	j.meshPath = filepath.Join(finalDir, "mesh.t2mesh")
	if len(input) > 0 {
		j.inputPath = filepath.Join(finalDir, "input.img")
	}
	if len(source) > 0 {
		j.sourcePath = filepath.Join(finalDir, "source.img")
	}
	j.previews = nil
	j.mu.Unlock()
	return nil
}

// restoreJobs scans only complete, versioned job directories. Temporary
// directories left by a killed write and malformed/corrupt entries are ignored
// with a log message; one bad asset must not prevent the server from starting.
func (s *server) restoreJobs() (int, error) {
	if s.storeDir == "" {
		return 0, nil
	}
	if err := os.MkdirAll(s.storeDir, 0o755); err != nil {
		return 0, fmt.Errorf("create job store: %w", err)
	}
	entries, err := os.ReadDir(s.storeDir)
	if err != nil {
		return 0, fmt.Errorf("read job store: %w", err)
	}
	restored := 0
	for _, entry := range entries {
		if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		dir := filepath.Join(s.storeDir, entry.Name())
		j, err := loadPersistedJob(dir)
		if err != nil {
			log.Printf("ignoring persisted job %s: %v", entry.Name(), err)
			continue
		}
		s.mu.Lock()
		if _, exists := s.jobs[j.ID]; !exists {
			s.jobs[j.ID] = j
			restored++
		}
		s.mu.Unlock()
	}
	return restored, nil
}

func loadPersistedJob(dir string) (*job, error) {
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var m persistedJob
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}
	if m.Version != persistedJobVersion {
		return nil, fmt.Errorf("unsupported manifest version %d", m.Version)
	}
	if m.ID == "" || filepath.Base(m.ID) != m.ID || m.ID != filepath.Base(dir) {
		return nil, fmt.Errorf("invalid job id %q", m.ID)
	}
	meshPath := filepath.Join(dir, "mesh.t2mesh")
	if st, err := os.Stat(meshPath); err != nil || !st.Mode().IsRegular() {
		if err == nil {
			err = fmt.Errorf("not a regular file")
		}
		return nil, fmt.Errorf("mesh: %w", err)
	}
	sourcePath := filepath.Join(dir, "source.img")
	if st, err := os.Stat(sourcePath); err != nil || !st.Mode().IsRegular() {
		sourcePath = "" // optional for generations saved before source retention
	}
	inputPath := filepath.Join(dir, "input.img")
	if st, err := os.Stat(inputPath); err != nil || !st.Mode().IsRegular() {
		inputPath = "" // legacy jobs stored their processed input as source.img
	}
	for i := range m.Frames {
		if st, err := os.Stat(filepath.Join(dir, persistedFrameName(i))); err != nil || !st.Mode().IsRegular() {
			if err == nil {
				err = fmt.Errorf("not a regular file")
			}
			return nil, fmt.Errorf("preview %d: %w", i, err)
		}
	}
	return &job{
		ID: m.ID, State: "done", CreatedAt: m.CreatedAt, StartedAt: m.StartedAt,
		FinishedAt: m.FinishedAt, DurationMS: m.DurationMS,
		Quality: m.Quality, Device: m.Device, Thumbnail: m.Thumbnail,
		PreviewSeq: len(m.Frames), Frames: m.Frames,
		LivePreview: m.LivePreview || len(m.Frames) > 0, StageTimings: m.StageTimings,
		pipeline: m.Pipeline, Seed: m.Seed, Steps: m.Steps,
		TextureSteps: m.TextureSteps, Guidance: m.Guidance,
		Background: m.Background,
		persistDir: dir, meshPath: meshPath, inputPath: inputPath, sourcePath: sourcePath,
	}, nil
}

func (s *server) loadJobMesh(j *job) (*meshData, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.mesh != nil {
		return j.mesh, nil
	}
	if j.meshPath == "" {
		return nil, fmt.Errorf("mesh not ready")
	}
	mesh, err := readMeshFile(j.meshPath)
	if err != nil {
		return nil, err
	}
	j.mesh = mesh
	return mesh, nil
}

func loadJobPreview(j *job, seq int) ([]byte, error) {
	j.mu.Lock()
	if seq < 0 || seq >= j.PreviewSeq {
		j.mu.Unlock()
		return nil, fmt.Errorf("no such preview frame")
	}
	if seq < len(j.previews) && j.previews[seq] != nil {
		blob := j.previews[seq]
		j.mu.Unlock()
		return blob, nil
	}
	dir := j.persistDir
	j.mu.Unlock()
	if dir == "" {
		return nil, fmt.Errorf("no such preview frame")
	}
	return os.ReadFile(filepath.Join(dir, persistedFrameName(seq)))
}

func writeMeshFile(path string, mesh *meshData) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0o644)
	if err != nil {
		return fmt.Errorf("create mesh: %w", err)
	}
	if err := writeMeshBinary(f, mesh); err != nil {
		f.Close()
		return fmt.Errorf("write mesh: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("sync mesh: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close mesh: %w", err)
	}
	return nil
}

func readMeshFile(path string) (*meshData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open mesh: %w", err)
	}
	defer f.Close()
	var magic [8]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, fmt.Errorf("read mesh magic: %w", err)
	}
	pbrWidth := 0
	switch string(magic[:]) {
	case "T2MESH01":
	case "T2MESH02":
		pbrWidth = 5
	case "T2MESH03":
		pbrWidth = 6
	default:
		return nil, fmt.Errorf("bad mesh magic %q", magic)
	}
	var nv32, nt32 uint32
	if err := binary.Read(f, binary.LittleEndian, &nv32); err != nil {
		return nil, fmt.Errorf("read vertex count: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &nt32); err != nil {
		return nil, fmt.Errorf("read triangle count: %w", err)
	}
	if nv32 == 0 || nt32 == 0 || nv32 > 100_000_000 || nt32 > 100_000_000 {
		return nil, fmt.Errorf("invalid mesh size %d vertices, %d triangles", nv32, nt32)
	}
	nv, nt := int(nv32), int(nt32)
	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat mesh: %w", err)
	}
	expected := int64(16) + int64(nv)*24 + int64(nv*pbrWidth)*4 + int64(nt)*12
	if st.Size() != expected {
		return nil, fmt.Errorf("mesh size is %d bytes, expected %d", st.Size(), expected)
	}
	m := &meshData{NVerts: nv, NTris: nt, Verts: make([]float32, 3*nv), Normals: make([]float32, 3*nv)}
	if err := binary.Read(f, binary.LittleEndian, m.Verts); err != nil {
		return nil, fmt.Errorf("read vertices: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, m.Normals); err != nil {
		return nil, fmt.Errorf("read normals: %w", err)
	}
	if pbrWidth != 0 {
		stored := make([]float32, pbrWidth*nv)
		if err := binary.Read(f, binary.LittleEndian, stored); err != nil {
			return nil, fmt.Errorf("read PBR attributes: %w", err)
		}
		if pbrWidth == 6 {
			m.PBR = stored
		} else {
			m.PBR = make([]float32, 6*nv)
			for i := 0; i < nv; i++ {
				copy(m.PBR[6*i:6*i+5], stored[5*i:5*i+5])
				m.PBR[6*i+5] = 1
			}
		}
	}
	m.Tris = make([]int32, 3*nt)
	if err := binary.Read(f, binary.LittleEndian, m.Tris); err != nil {
		return nil, fmt.Errorf("read triangles: %w", err)
	}
	var extra [1]byte
	if n, err := f.Read(extra[:]); err != io.EOF || n != 0 {
		return nil, fmt.Errorf("mesh contains trailing data")
	}
	return m, nil
}
