# AGENTS.md – trellis2.cpp

**trellis2.cpp** – a C++/[ggml](https://github.com/ggml-org/ggml) port
of the **TRELLIS.2** Image-to-3D pipeline: image in, 3D mesh with
per-vertex PBR out, the entire inference in C++/ggml (no PyTorch at
runtime). A Go demo server with a browser viewer drives the pipeline
through a flat C-ABI.

- **Origin:** this is a downstream fork, not an original work. The
  initial stage-1 geometry port is by rms80 (author of
  [sam3.cpp](https://github.com/rms80/sam3.cpp), on which this project is
  structurally modeled); the bulk of the pipeline — DINOv3 encoder,
  shape-SLAT stage, 1024 cascade, PBR texturing, GLB export, print
  remeshing, and the Go demo server — is by Richard Palethorpe
  (`richiejp` remote, upstream). Validation process after
  depth-anything.cpp, fuzzing process after privacy-filter.cpp.
- **This fork:** Robert Beckebans — Windows/HIP/ROCm and Vulkan support,
  the DLL build for the Go server, block-split decoding of the finest
  decoder level, and the ggml 0.19 upgrade. Do not attribute upstream
  work to this fork; check `git log`/`git blame` before claiming who
  designed a given path.
- **Language:** C++14 (C++17 only in the optional CGAL path), Go (demo
  server), Python (converter + PyTorch reference) — **code comments in
  English**
- **Architecture:** single-file library (`src/trellis2.h` / `src/trellis2.cpp`)
  with bundled ggml submodule, DLL export decoration and flat C-ABI
  (`trellis2_capi.h`); examples, tests, and the Go server are pure
  consumers of this library.
- **Goal:** image → mesh without a Python runtime, numerically
  tap-for-tap validated against the PyTorch reference, up to the 1024³
  cascade on a 16-GB GPU.

---

## Workflow for AI agents

These rules are binding and complement the technical architecture
rules below.

### Read before every larger change

1. `AGENTS.md` (this file)
2. The relevant architecture entry under `docs/architecture/README.md`
3. The matching plan under `docs/plan/` or the roadmap plan
   `docs/PLAN.md`
4. `docs/VERIFICATION.md`, as soon as a neural stage path is touched
5. Relevant affected files in the current working tree

Do not work from chat memory — read the real files first.
`src/trellis2.cpp` is large (>170 kB); read the affected stage function
specifically instead of skimming the whole file.

### Plans and phases

For larger features, bugs with unclear code path, and architecture
changes, a plan is created first or a matching existing roadmap plan is
reused.

New issue-based plans use:

```text
docs/plan/<github-issue>-<short-name>.md
```

- Issue numbers come from
  <https://github.com/RobertBeckebans/AI_trellis2cpp/issues>.
- If no issue exists yet but the existing roadmap plan `docs/PLAN.md`
  contains the affected section (stage porting, validation, fuzzing,
  demo), no artificial issue must be created. The plan key is then
  `plan`.
- Template: `docs/plan/_TEMPLATE.md`.
- Conventions: `docs/plan/README.md`.
- Small trivial changes like typos or obvious one-line fixes do not
  need a plan.

After completing larger plan phases, a progress entry is prepared. If
a GitHub issue exists, the issue number is used:

```text
docs/progress/<github-issue>_<phase>-<short-name>.md
```

If no issue exists yet but an existing plan or roadmap section is
affected, the plan key is used instead:

```text
docs/progress/<plan-key>_<phase>-<short-name>.md
```

Example: `docs/PLAN.md`, phase "texture stage" →
`docs/progress/plan_texture-stage.md`.

- Template: `docs/progress/_TEMPLATE.md`.
- Conventions: `docs/progress/README.md`.
- Since AI agents do not commit, `Commits: <to be added after review>`
  remains in place until the reviewer commits after review.

### Git and review workflow

- AI agents **never commit themselves** and never push.
- Changes are only prepared so the reviewer can review them against the
  current Git state.
- Do not run any write Git commands unless the user explicitly requests
  it: no `git add`, `git commit`, `git push`, `git reset`,
  `git restore`, `git checkout`, `git switch`, `git rebase`, `git merge`.
- Read-only Git commands are allowed when they help with review:
  `git status`, `git diff`, `git log`, `git show`.
- The repository has multiple remotes (`origin` = internal Git server,
  `github` = `RobertBeckebans/AI_trellis2cpp`, `richiejp` = upstream
  fork). Never select or synchronize a remote on your own.
- The `ggml/` submodule is its own repository
  (`RobertBeckebans/ggml`, tracking upstream ggml). Changes to it do
  not belong in a trellis2 commit; a submodule pointer update is a
  deliberate decision to be discussed separately.

### Destructive filesystem commands

- **Never** run `rm -rf`, `rm -r`, `rmdir /s`, `del /s`,
  `Remove-Item -Recurse` or comparable recursive-delete commands without
  first presenting the exact deletion path to the user and obtaining
  explicit approval. A single overwrite error in the path can delete
  entire trees that are listed in `.gitignore` and therefore not
  recoverable via Git.
- Especially at risk and **not** recoverable via Git here:
  `ggufs/` (~14 GB converted weights), `models/` (~7 GB
  HF Safetensors), `dumps/` (reference activations from the PyTorch
  container), `generations/` (persisted server jobs), `assets/`,
  `docs/ideas/`, `docs/bugs/`, `docs/screenshots/`.
- The existing build scripts (`cmake-ninja-win64-*.bat`,
  `cmake-vs2022-win64.bat`) themselves contain `rmdir /s /q build`.
  They may only be started after explicit user request; for own builds
  create a separate `build-*` directory instead of throwing away an
  existing one.
- The same applies to `mv`/`move`/`rename` over existing files and to
  `git clean -fd`, `del CMakeCache.txt` in foreign build trees, or any
  operation that potentially overwrites or removes a directory or file
  **outside the test sandbox currently being created by the agent**.
- Test artifacts that the AI agent itself created (e.g. a
  `test_sandbox/` folder for a smoke test) may be deleted, **but only
  with a path that names exactly this sandbox folder** — never a parent
  directory as a whole, and never with wildcards like `ggufs/*` or
  `generations/*` that could accidentally hit user data.
- Before each delete command: formulate the exact command **with full
  path** as a proposal, wait for confirmation, then execute. The same
  applies to `git rm`, `git clean`, `docker system prune`.
- If cleaning up a test run would require deleting more than the
  self-created sandbox folders: leave the state as is and list the
  resulting artifacts (paths, sizes) to the user — cleanup is then a
  human decision, not an AI task.

- At the end of a completed work phase, list:
  1. changed files,
  2. short summary,
  3. checks that ran with their result,
  4. open points or risks,
  5. proposed English commit message.

### Language and license rules

- Documentation and plan/progress entries: English.
- Code comments and identifiers: English.
- Git logs and proposed commit messages: English, matching the
  existing repository history (concise, descriptive sentences in
  present/imperative mood, no Conventional-Commits prefix).
- License: MIT — see [`LICENSE`](LICENSE).
- Vendored third-party code under `third_party/` is also MIT or
  Public Domain: `meshoptimizer`, `xatlas`, `tinybvh`, `stb`, and the
  `autoremesher` core with its bundled `isotropicremesher`. New vendor
  dependencies only with MIT/BSD/Azure/PD-compatible license. Every
  component is itemised in [`THIRD_PARTY.md`](THIRD_PARTY.md), which is
  updated in the same change that adds or removes one.
- **Eigen / MPL-2.0 exception:** the optional quad remesh backend
  (`src/quad_remesh.cpp`, `third_party/autoremesher/`) needs Eigen 5.x,
  which is MPL-2.0 and therefore **not** vendored. It is declared in
  `vcpkg.json` and consumed through `find_package(Eigen3 CONFIG)` →
  `Eigen3::Eigen`, so no MPL2 file enters this MIT repository. MPL-2.0
  is file-level copyleft and explicitly permits combination with a
  differently licensed Larger Work (§3.3); using it unmodified through
  a package manager discharges the remaining duty. A build without
  Eigen 5.x self-disables the option at configure time.
- **CGAL exception:** the optional alpha-wrap backend
  (`src/print_remesh.cpp`) links against CGAL ≥ 5.5. CGAL's 3D alpha
  wrapping is GPL-3.0-or-later. Therefore this path stays strictly
  encapsulated behind `TRELLIS2_CGAL` / `TRELLIS2_USE_CGAL=1` and must
  not leak into the standard build: no CGAL headers or types in
  `src/trellis2.h`, `src/trellis2_capi.h`, `src/mesh_export.h`, or in
  unconditionally compiled code paths.
- No adoption of code from third-party projects whose license is
  incompatible with the project (in particular no GPL/AGPL/LGPL code
  in the MIT library). Behavior, file formats, and algorithms must
  only be implemented in a license-safe manner under own design — this
  explicitly applies to the original TRELLIS reference as well: what is
  reproduced is the *behavior* (tensor layouts, hyperparameters,
  numerical results), not the source text.
- Feedback, cross-check, audit, or idea is not a build order unless an
  explicit Go follows.

### Toolchain setup

**C++/CMake** is the primary toolchain. There is no project-local
language environment that arises automatically — compiler, CMake,
Ninja, and optionally ROCm/CUDA/Vulkan SDK come from the system.

- Submodule first: `git submodule update --init --depth 1` (ggml).
- Windows: the prepared batch scripts in the root pick generator and
  backend (see *Build & Run*). They are the author's reference path.
- Linux/CUDA: via the containers in `docker/` (`Dockerfile.demo` for
  the demo build, `Dockerfile.ref` for the PyTorch reference).

**Python** is used exclusively for weight conversion and reference
dumps and runs **in the reference container**, not on the host:

- `docker build -f docker/Dockerfile.ref -t trellis2-ref docker`
- `docker run --rm -v "$PWD":/work -w /work trellis2-ref python scripts/convert_*.py …`
- Do not suggest system `pip install` on the host; do not introduce
  new host dependencies without anchoring them in the Dockerfile.

**Go** (demo server, `server/`): standard toolchain, dependencies in
`server/go.mod`. The server loads `libtrellis2.so`/`.dll` via dlopen —
signature changes to the C-ABI must be propagated there.

**Formatting**: `clang-format` **18.1.8**, driven via `format_code.sh`
(or `format_code.bat` on Windows). Never run `clang-format` directly
with a different version — that produces project-wide diff noise.

```sh
./format_code.sh          # headers with .clang-format-header, sources with .clang-format-cpp
```

The script temporarily copies the matching config to `.clang-format`
and deliberately excludes `ggml/`, `third_party/`, `build*/`. Vendored
code and the ggml submodule are **never** reformatted.

Further helper tools:

```sh
tools/count_loc.bat          # cloc statistics without build/target/docs/assets
tools/yek_collect_repo.bat   # repo skeleton for context collection into .yek/
```

---

## Library design (own design)

trellis2.cpp deliberately follows the sam3.cpp / llama.cpp pattern: a
compact library with a stable, decorated API instead of a framework.

### Naming conventions

- Public C++ API: prefix `trellis2_` (`trellis2_dino_encode`,
  `trellis2_ss_flow_sample`, `trellis2_shape_dec_decode`), structs
  `trellis2_<thing>`.
- Flat C-ABI: prefix `t2_` for functions, `T2_` for enums/constants
  (`t2_generate`, `T2_STAGE_SLAT_FLOW`, `T2_CAPI_ABI_VERSION`).
- Every exported C++ function carries `TRELLIS2_API`, every C-ABI
  function `TRELLIS2_CAPI`. If the decoration is missing, the shared
  library build breaks (or worse: only the dlopen of the Go server at
  runtime).
- CMake options: prefix `TRELLIS2_` (`TRELLIS2_BUILD_TESTS`,
  `TRELLIS2_CGAL`, `TRELLIS2_FUZZ`, `TRELLIS2_USE_EXTERNAL_GGML`).
- Internal helpers stay in anonymous namespaces or `static` — do not
  export anything new into the global namespace.

### Stage structure

Every pipeline stage is an independently callable pair of **loader**
(`*_load`, reads the GGUF file including `trellis2.<stage>.*` KV
metadata) and **forward** (`*_forward` / `*_sample` / `*_decode`,
builds the ggml graph). This is the prerequisite for each stage to be
tappable individually against the PyTorch reference — please create
new stages in the same pattern.

### Backend selection

The first GPU backend exposed by ggml is chosen (CUDA / HIP / Metal /
Vulkan), with CPU fallback. No backend may be hardwired in the code;
backend-specific branches belong behind a feature/capability query,
not behind `#ifdef` on the build type.

### Numerical compatibility

`f16` is the delivery path, `f32` the validation path (`--ftype 0`).
Changes to a stage must keep the parity table in
`docs/VERIFICATION.md` valid; if a tap worsens, that is a bug, not a
tolerance problem.

## Directory structure

```
trellis2.cpp/
├── src/                             # the library itself
│   ├── trellis2.h / trellis2.cpp    # all stages, loaders, samplers
│   ├── trellis2_capi.h / .cpp       # flat C-ABI for the Go server (ABI version!)
│   ├── mesh_export.h / .cpp         # GLB export: vertex PBR or UV atlas (xatlas)
│   ├── print_remesh.h / .cpp        # optional: CGAL alpha wrap + PBR transfer
│   ├── quad_remesh.h / .cpp         # optional: AutoRemesher quad retopology
│   └── pbr_utils.h                  # shared PBR helpers
├── CMakeLists.txt                   # library, options, CGAL probe
├── examples/                        # CLI tools, consumers of the library
│   ├── dino_info, dino_encode       # inspect / produce conditioning
│   ├── ss_flow_info, ss_sample      # stage-1 DiT: metadata / sampling
│   ├── ss_decode, ss_mesh           # occupancy decoder / coarse preview mesh
│   ├── mesh2glb                     # .t2mesh → GLB (also --print)
│   ├── marching_cubes.h             # Freudenthal isosurface, single header
│   └── flexible_dual_grid.h         # dual grid → triangle mesh
├── tests/                           # ctest targets, parity against PyTorch
│   ├── parity.hpp                   # atol+rtol gate, reporting
│   ├── test_*.cpp                   # per stage or per invariant
│   └── ref_*.py                     # reference generators (in ref container)
├── fuzz/                            # libFuzzer harnesses (clang, ASan/UBSan)
├── server/                          # Go demo server + embedded web viewer
│   └── web/                         # self-contained WebGL viewer (no CDN/build)
├── scripts/                         # converters, reference dumps, downloads, demo
├── docker/                          # Dockerfile.demo (CUDA+Go), Dockerfile.ref (PyTorch)
├── third_party/                     # vendored: xatlas, meshoptimizer, tinybvh, autoremesher, stb
├── ggml/                            # submodule (RobertBeckebans/ggml)
├── tools/                           # small helper tools (cloc, yek)
└── docs/
    ├── PLAN.md                      # roadmap: porting status of the pipeline
    ├── VERIFICATION.md              # parity table + reference workflow
    ├── architecture/                # architecture index
    ├── ideas/                       # idea collection (not a build order)
    ├── bugs/                        # known bugs, optionally with screenshots
    ├── screenshots/                 # development screenshots
    ├── progress/                    # progress tracking (per phase)
    └── plan/                        # planning documents
```

Not versioned (gitignored, large, regenerable, or private):
`models/`, `ggufs/`, `dumps/`, `generations/`, `assets/`, `build*/`.

## Build & Run

### Primary path: Windows + HIP/ROCm

Development happens on Windows against a **Radeon AI PRO R9700**
(RDNA4, `gfx1201`) via HIP/ROCm — that is the fastest backend available
here and the configuration everything is expected to work in. Assume this
path when reasoning about performance, VRAM, or backend behavior, unless
the task explicitly concerns another backend.

```bat
cmake-ninja-win64-rocm.bat        :: THE build: HIP/ROCm, Ninja Multi-Config,
                                  ::   shared lib -> build\bin\Release\libtrellis2.dll
                                  ::   (+ ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-hip.dll)
cmake-ninja-win64-rocm-cgal.bat   :: same, plus the CGAL alpha-wrap backend
start_server.bat                  :: run the Go demo server against that DLL
```

What `cmake-ninja-win64-rocm.bat` pins, so it does not have to be
rediscovered: ROCm at `C:\Program Files\AMD\ROCm\6.4`, its `clang`/
`clang++` as the compilers, `GGML_HIP=ON` with `GGML_VULKAN=OFF`,
`AMDGPU_TARGETS=gfx1201`, `hipblas_DIR` from the ROCm tree,
`BUILD_SHARED_LIBS=ON`, `TRELLIS2_BUILD_TESTS=ON`,
`TRELLIS2_BUILD_EXAMPLES=OFF`, and an **absolute**
`CMAKE_RUNTIME_OUTPUT_DIRECTORY` (relative paths get reinterpreted
per-subdirectory by ggml's nested CMakeLists).

Caveat: the script starts with `rmdir /s /q build` and therefore throws
away the existing build tree. Only run it on explicit request; for
experiments configure a separate `build-*` directory by hand with the
same flags.

### Secondary Windows paths

```bat
cmake-ninja-win64-vulkan.bat      :: Vulkan backend
cmake-ninja-win64-cpu.bat         :: pure CPU build
cmake-vs2022-win64.bat            :: generate a Visual Studio project
```

### Portable / CI paths

```sh
git submodule update --init --depth 1     # ggml

# standard build (static lib + examples)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# with tests
cmake -B build -DTRELLIS2_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build -LE model          # fast, without assets
ctest --test-dir build                    # full parity (needs ggufs/ + dumps/)

# shared library for the Go server (Linux/CUDA, e.g. inside docker/Dockerfile.demo)
cmake -B build-shared -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON && cmake --build build-shared -j
cd server && go build -o trellis2-server .

# fuzzing (clang)
cmake -B build-fuzz -DTRELLIS2_FUZZ=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

Important CMake options:

| Option | Default | Effect |
|---|---|---|
| `TRELLIS2_AUTOREMESHER` | `ON` | quad-remesh probe; without Eigen 5.x simply inactive |
| `TRELLIS2_BUILD_EXAMPLES` | `ON` | CLI examples in `examples/` |
| `TRELLIS2_BUILD_TESTS` | `OFF` | ctest targets in `tests/` |
| `TRELLIS2_CGAL` | `ON` | alpha-wrap probe; without CGAL ≥ 5.5 simply inactive |
| `TRELLIS2_FORCE_TINYBVH` | `OFF` | PBR projection via tinybvh even when CGAL is present |
| `TRELLIS2_FUZZ` | `OFF` | libFuzzer + ASan/UBSan, globally instrumented |
| `TRELLIS2_METAL` | `ON` | Metal backend on Apple |
| `TRELLIS2_USE_EXTERNAL_GGML` | `OFF` | ggml via `find_package` instead of submodule |
| `BUILD_SHARED_LIBS` | `OFF` | sets `TRELLIS2_SHARED`/`TRELLIS2_BUILD` |

## `src/trellis2.h` / `src/trellis2.cpp` — the library

The entire inference path. Entry points per stage:

- `trellis2_remove_solid_background_rgba()` — deterministic background
  removal (edge-connected black/white → feathered alpha).
- `trellis2_preprocess_rgba()` — alpha-bbox crop, premultiply,
  PIL-exact Lanczos-512. **Byte-exact** against Pillow; any change
  here invalidates all downstream references.
- `trellis2_dino_encode()` — DINOv3 ViT-L/16, produces the
  `[1, 1029, 1024]` conditioning with affine-free final LayerNorm.
- `trellis2_load_dinodata()` — `.dinodata` reader for CLI chaining and
  tests.
- `trellis2_ss_flow_load()` / `_forward()` / `trellis2_ss_flow_sample()`
  — dense 1.3B DiT, 12-step CFG flow Euler → `z_s`.
- `trellis2_ss_dec_decode()` — dense 3D-conv decoder → 64³ occupancy →
  32³ voxel scaffold.
- `trellis2_slat_flow_sample()` — sparse 1.3B DiT over the active
  voxels.
- `trellis2_shape_dec_decode()` — `FlexiDualGridVaeDecoder`, sparse
  ConvNeXt U-Net, 16× upsampling (32³ → 512³).
- Texture path: shape encoder → texture SLAT flow → guided decoder →
  trilinear PBR sampling at the dual-grid surface points.

## `src/trellis2_capi.h` / `.cpp` — the C-ABI

The only interface of the Go server. Contains `T2_CAPI_ABI_VERSION`,
the stage enums for the progress callback, and the pipeline types.

**Rule:** every signature or struct-layout change here bumps
`T2_CAPI_ABI_VERSION` and must be propagated in `server/engine.go`. A
forgotten bump manifests as a silent memory error at runtime, not as
a compile error.

## `src/mesh_export.{h,cpp}` / `src/print_remesh.{h,cpp}` / `src/quad_remesh.{h,cpp}` — export

- GLB export without GPU: dense materials as interpolated vertex
  colors, metallic/roughness additionally in the custom attribute
  `_METALLIC_ROUGHNESS`; optional xatlas UV atlas.
- `print_remesh` is the optional CGAL path (watertight alpha wrap +
  closest-surface PBR rebake). The alpha wrap is fully behind
  `TRELLIS2_USE_CGAL`; the closest-surface projection is not, since it
  also has a tinybvh backend and is compiled unconditionally.
- `quad_remesh` is the optional AutoRemesher path (quad-dominant
  mid-poly retopology), behind `TRELLIS2_AUTOREMESHER` and MIT
  throughout. It is **not** a replacement for the wrap: it guarantees
  nothing about closedness and reports `boundary_edges` so callers can
  say so. The two compose — see
  [`docs/architecture/quad-remesh.md`](docs/architecture/quad-remesh.md).

## `server/` — Go demo server

stdlib HTTP, a single inference mutex, `POST /api/generate` → job id,
progress polling, `GET /api/mesh/<id>`, persistent jobs under
`generations/` (atomically written, incomplete writes are ignored at
startup), `-unload-idle` for VRAM release. The web viewer in
`server/web/` is deliberately self-contained: **no CDN, no build step**
— do not give up this property.

## Tests & validation

The validation process is the core of the project, not an appendix.

- Reference activations are dumped in the `trellis2-ref` container
  (`scripts/dump_*_reference.py`, `scripts/refgen.sh`) and stored as
  GGUF under `dumps/`.
- The C++ comparison runs via `tests/parity.hpp` with the gate
  `|got-ref| <= atol + rtol*|ref|` (default 2e-3), reported as
  max-abs / rel-L2 / per-row.
- Tests without assets (marching cubes, preprocess, background removal,
  PBR sampling, chunked decode) always run; model tests use
  `SKIP_RETURN_CODE 77` when fixtures are missing — a skip is **not** a
  passing test and must not appear as such in the report.
- New stage ports need their own tap test plus an entry in the parity
  table of `docs/VERIFICATION.md`.

## Fuzzing

libFuzzer harnesses in `fuzz/` cover the untrusted inputs: image bytes
(stb_image decode + preprocess, the demo upload path) and the
`.dinodata` loader. GGUF model files are considered trusted assets.
Reproduced crashes are stored under `fuzz/crashes-fixed/` when they
are fixed.

## Conventions: coordinates & data formats

- Meshes live in the centered unit cube; `--normalize` in `ss_mesh`
  establishes this explicitly.
- Resolution levels: 64³ occupancy → 32³ scaffold → 512³ (fine) →
  optional 1024³ or 1536³ cascade. The 1536 tier reuses the 1024
  checkpoints and can be stepped back down toward 1024 by the HR token
  budget (`src/cascade_tokens.h`); it is implemented but not yet
  measured end to end (`docs/plan/1536-cascade.md` phase 3).
- Own container formats: `.dinodata` (conditioning), `.latent` (z_s),
  `.occ` (occupancy logits), `.t2mesh` (`T2MESH01/02/03`, versioned).
  Format changes require a plan and need a version bump plus a read
  path for the old versions.
- GGUF hyperparameters are stored as KV metadata under
  `trellis2.<stage>.*`; weights keep their original checkpoint names so
  that converter and loader remain independently checkable.