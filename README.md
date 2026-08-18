# trellis2.cpp

A C++/[ggml](https://github.com/ggml-org/ggml) implementation of the
**TRELLIS.2** image-to-3D pipeline: an image goes in, a 3D mesh with per-vertex
PBR textures comes out, with all inference in C++/ggml (no PyTorch at runtime).
The demo can also export the result into a portable, full-density **GLB** with
standard interpolated vertex colour and retained PBR attributes — no reference
container required.

Modeled structurally after [sam3.cpp](https://github.com/rms80/sam3.cpp):
single-file library (`src/trellis2.h` / `src/trellis2.cpp`), bundled ggml as a
submodule (Metal on by default on Apple), DLL-export decoration, and a CMake
build with example executables. A flat C ABI (`src/trellis2_capi.h`) drives a Go
demo server with a browser mesh viewer.

> **How it works** — the pipeline stage by stage, how the code is layered, the
> data formats, and how correctness is established:
> **[`docs/architecture/README.md`](docs/architecture/README.md)**.
> Tap-by-tap parity numbers: [`docs/VERIFICATION.md`](docs/VERIFICATION.md).
> Rules and conventions for contributors: [`AGENTS.md`](AGENTS.md).

## What this fork adds

This is a **downstream fork**. The port itself is upstream work: the stage-1
geometry port by [rms80](https://github.com/rms80), and the complete image→mesh
pipeline — DINOv3 encoder, shape-SLAT stage, demo server, PBR texturing, CGAL
print wrap — by [Richard Palethorpe](https://github.com/richiejp), through
`84f4a2c`. Everything below was added on top of that base; check `git log`
before attributing a design decision.

- **Native Windows on four backends.** Upstream reaches the GPU through a Linux
  CUDA container. This fork adds a Windows DLL build with the Go server beside
  it, and configure scripts for CPU, **CUDA**, Vulkan and **HIP/ROCm** —
  developed on a Radeon AI PRO R9700 (`gfx1201`), a backend that did not exist
  here before. CUDA is *not exercised here* for want of the hardware.
- **Two GPU backends in one build, switched at runtime** via
  `TRELLIS2_DEVICE=cpu|rocm|vulkan` or a dropdown in the viewer — no second
  build tree, no restart. The backend is recorded in each generation's manifest.
- **1536³ cascade tier**, with a token budget that steps the resolution back
  down when the scaffold would exceed it rather than failing at the decode.
- **Quad retopology and a normal-map bake**, so generated geometry is usable
  downstream: the [AutoRemesher](https://github.com/huxingyi/autoremesher) core
  vendored without Qt or TBB, plus a **MikkTSpace** tangent-space normal map
  baked into the GLB (via `meshopt_generateTangents`) — the basis glTF 2.0
  prescribes, and therefore the only one that stays correct across conformant
  importers. See [`docs/plan/autoremesher-quad-remesh.md`](docs/plan/autoremesher-quad-remesh.md).
- **A smaller GPL surface.** Closest-surface PBR projection moved from CGAL to
  [tinybvh](https://github.com/jbikker/tinybvh), so a default configure produces
  generations and full-density GLB exports on MIT terms alone. The *low-poly*
  path still needs CGAL's Alpha Wrap — see the licence section.
- **Two silent GPU defects found, isolated and worked around.** On `gfx1201`
  `mul_mat` stops writing past a fixed column count and still reports success,
  which sheared a quarter off every 1024³ mesh with nothing logged; and flash
  attention accumulates in F16 regardless of `GGML_PREC_F32`, which cost thin
  structure at the cascade tiers. Both are reproduced, documented and mitigated
  — see [`docs/bugs/`](docs/bugs/) and the architecture document's *Backends and
  their sharp edges*.
- **Numerical parity you can actually run.** The PyTorch reference environment
  is reproducible natively through `uv`, including AMD's Windows ROCm wheels, so
  the golden dumps no longer need the CUDA container. The reference can also
  produce a *whole generation* that lands in the viewer beside the backend runs
  of the same image ([`scripts/ref_generate.py`](scripts/ref_generate.py)).
- **Diagnosability throughout**: per-stage timings, a progress trace, and
  warnings when the decoder's subdivision runs away *or* collapses — the latter
  because the worst failure on record passed the one-sided check in silence.

## Quick start (Windows, AMD)

The primary path for this fork. No container.

```sh
git clone --recursive https://github.com/RobertBeckebans/AI_trellis2cpp.git
cd AI_trellis2cpp
scripts/download_ggufs.sh          # prebuilt f16 GGUFs -> ggufs/ (~14 GB)
cmake-ninja-win64-rocm.bat         # HIP + Vulkan in one library
start_server.bat
# open http://localhost:8742 and drop an image
```

Three things worth knowing before the first run:

- `scripts/download_ggufs.sh` is a shell script — run it from **Git Bash**, not
  from `cmd`.
- `cmake-ninja-win64-rocm.bat` needs the **ROCm SDK** and the **Vulkan SDK** (it
  refuses to start without `VULKAN_SDK`, since it builds both backends into one
  library). To skip Vulkan, set `-DGGML_VULKAN=OFF`, or use
  `cmake-ninja-win64-cpu.bat` / `cmake-ninja-win64-vulkan.bat`. It **wipes
  `build/`**, so it is a configure step, not a rebuild step.
- `start_server.bat` is the normal way to start: it rebuilds whatever is stale
  and puts the DLL and the ROCm runtime on `PATH` before launching. That `PATH`
  also matters for `ctest`, which otherwise blocks in the loader.

## Quick start (Windows, NVIDIA)

CUDA no longer needs the Linux container.

```sh
git clone --recursive https://github.com/RobertBeckebans/AI_trellis2cpp.git
cd AI_trellis2cpp
scripts/download_ggufs.sh          # prebuilt f16 GGUFs -> ggufs/ (~14 GB)
cmake-msbuild-win64-cuda.bat       # CUDA + Vulkan in one library, into build-cuda\
start_server.bat build-cuda
# open http://localhost:8742 and drop an image
```

Needed once, beyond the AMD list: **CUDA Toolkit 12.8+** (`CUDA_ARCHS` at the
top of the script defaults to `120` / Blackwell — use `89` for RTX 40xx, `86`
for 30xx, `75` for 20xx), **Visual Studio 2019/2022 or the Build Tools** (nvcc
only accepts MSVC as host compiler on Windows), and **vcpkg with `eigen3` and
`cgal`**, expected at `C:\vcpkg`. Without vcpkg the print wrap and quad stage
configure as unavailable rather than failing, so read the configure output.

It builds into `build-cuda\`, so AMD and NVIDIA trees can coexist.

**This path is untested here** — there is no NVIDIA card in this machine. The
script is modelled on the working ROCm one and checked against ggml's CMake.
CUDA does inherit one change: ggml's CUDA graphs are disabled by default because
they crash on HIP and the two share that code. `TRELLIS2_CUDA_GRAPHS=1` re-enables
them.

## Quick start (docker, CUDA)

The upstream container path, unchanged. Linux and an NVIDIA card.

```sh
git submodule update --init --depth 1                 # ggml
scripts/download_ggufs.sh                             # prebuilt f16 GGUFs -> ggufs/ (~14 GB)
docker build -f docker/Dockerfile.demo -t trellis2-demo docker   # CUDA runtime + Go
docker run --rm -v "$PWD":/work -w /work trellis2-demo bash -c '
  cmake -B build-cuda-shared -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SHARED_LIBS=ON && cmake --build build-cuda-shared -j
  cd server && go build -o trellis2-server-linux .'
docker run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work/server -p 8742:8742 \
  trellis2-demo ./trellis2-server-linux -lib /work/build-cuda-shared/libtrellis2.so \
  -ggufs /work/ggufs -store /work/generations -unload-idle
# open http://localhost:8742 and drop an image
```

Or run `scripts/demo.sh`, which builds the lib + server, downloads missing
GGUFs, and launches the container.

## Prebuilt GGUFs

`scripts/download_ggufs.sh` pulls ready-made f16 GGUFs from three public repos
under the [LocalAI-io](https://huggingface.co/LocalAI-io) org, so the
safetensors download and the conversion step can be skipped entirely:

- [`TRELLIS.2-4B-GGUF`](https://huggingface.co/LocalAI-io/TRELLIS.2-4B-GGUF) — MIT
- [`TRELLIS-image-large-GGUF`](https://huggingface.co/LocalAI-io/TRELLIS-image-large-GGUF) — MIT
- [`dinov3-vitl16-pretrain-lvd1689m-GGUF`](https://huggingface.co/LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF) — DINOv3 License (Built with DINOv3)

**Developers** who need the f32 validation variants, or who want to regenerate
the GGUFs from source, use `scripts/download_models.sh` (HF safetensors →
`models/`, ~7 GB) then `scripts/convert_all.sh`. Conversion runs with or without
the container — see [`docs/reference-environment.md`](docs/reference-environment.md),
the same `uv` environment the parity dumps use. Card and licence sources for the
published repos live in `scripts/hf/`.

## The demo server

Completed generations are committed atomically under `generations/` (mesh,
replay frames, manifest) and restored with the same job IDs after a restart.
`-store ''` disables persistence, `-store PATH` relocates it, incomplete writes
are ignored on startup. With `-unload-idle` the server starts without allocating
model VRAM, loads on the first generation, and releases when the queue is idle.

In the browser:

- **Quality** — coarse preview (64³ marching cubes), 512³ fine, 1024³ cascade
  (the TRELLIS.2 default), or 1536³. Coarse falls back automatically if the
  shape-SLAT models are absent (`-coarse`); both cascades need the extra 1024
  model (`-no-1024` disables them).
- **Compute device** — CPU, ROCm or Vulkan from the same build; switching drops
  the resident models so the next generation reloads on the chosen backend.
- **Attention** — `auto`, `exact` or `flash`. Exact costs roughly 2× the wall
  clock and is what keeps thin structure intact at 1024 and above; see
  [`docs/progress/backend-parity_1024-exact-attention.md`](docs/progress/backend-parity_1024-exact-attention.md).
- **Live steps** — off by default, because each sparse-structure frame needs an
  extra CPU occupancy decode between GPU steps. Enabling it records the frames
  used by replay and showcase mode.
- **Asset export** — preserves the generated polygon count, previews component
  cleanup, optionally keeps only the largest piece, and downloads a
  Three.js-ready GLB. Dense materials are stored as standard interpolated vertex
  colours rather than a sub-texel atlas; metallic/roughness are retained in a
  custom `_METALLIC_ROUGHNESS` attribute. All components are preserved unless
  destructive cleanup is selected explicitly. Engine-facing switches: base-colour
  only, opaque (no alpha link in Blender), and a unit scale.

Per-job `durationMs`, `livePreview` and `stageTimings` are exposed through
`/api/job/{id}` and persisted in the manifest, along with the sampling
parameters and the effective run configuration — so a stored generation says
what produced it. `/showcase` is a separate full-screen storyboard that replays
recorded stages and lingers on a rotating final model. **Regenerate from saved
image** re-runs a stored generation with the current settings.

Built with CGAL 5.5+, export also offers **watertight print wrap**: CPU Alpha
Wrap over the selected components, previewed before download. CGAL guarantees a
closed, oriented, intersection-free, 2-manifold result. `detail size` controls
the smallest cavities the wrap enters, `offset` how tightly it encloses — both
percentages of the source bounding-box diagonal. Because wrapping creates new
vertices the browser preview is geometry-only; GLB download then unwraps with
xatlas and rebakes base colour, metallic, roughness and opacity by projecting
each atlas texel through a CGAL AABB tree onto the closest source triangle. This
fixes topology only — physical scale, wall thickness and supports still need
choosing for the target printer. Also available without the server:

```sh
./build/examples/mesh2glb mesh.t2mesh printable.glb --print 1 0.0333
# percentages: alpha/detail size, then enclosing offset
```

On a 16 GB RTX 50-series: 512 fine runs image→mesh in ~110 s (~1M vertices); the
1024 cascade adds a second 1.3B pass and the 1024³ decoder for ~5M vertices
(~5 min, ~10 GB VRAM, ~14 GB host-RAM spike for the sparse-conv decode).

## Pipeline and CLI tools

The full stage-by-stage walkthrough, with diagrams, is in
[`docs/architecture/README.md`](docs/architecture/README.md). In short: image →
DINOv3 conditioning → sparse-structure DiT → 64³ occupancy → shape-SLAT DiT →
sparse ConvNeXt U-Net decoder (16× up to 512³) → dual-grid mesh, with an
optional 1024³/1536³ cascade and a texture branch producing 6-channel PBR.

Every stage is also a standalone CLI, which is how the tests and the reference
workflow drive them:

| tool | what |
|---|---|
| `dino_info`, `dino_encode` | inspect / produce the conditioning tokens |
| `ss_flow_info`, `ss_sample` | stage-1 DiT metadata / sampling → `.latent` |
| `ss_decode`, `ss_mesh` | occupancy decode / coarse marching-cubes preview |
| `dual_grid_cli` | 7-channel decoder fields → triangle mesh |
| `mesh2glb` | `.t2mesh` → GLB (also `--print`, `--quad`) |
| `quad_remesh_cli` | quad-dominant mid-poly retopology |

```sh
./build/examples/dino_encode ggufs/dino_f16.gguf image.png cond.dinodata
./build/examples/ss_sample   ggufs/ss_flow_f16.gguf cond.dinodata z_s.latent
./build/examples/ss_mesh     ggufs/ss_dec_f16.gguf  z_s.latent shape.obj --normalize
```

## Validation

Every neural stage is gated tap-by-tap against a PyTorch f32 reference, with
separate integration regressions for subdivision guidance, sparse material
sampling and GLB alpha preservation. The table, the tolerances, the device each
reference was produced on, and the commands to regenerate any of it are in
**[`docs/VERIFICATION.md`](docs/VERIFICATION.md)**.

```sh
cmake -B build -DTRELLIS2_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build -C Release -LE model     # no assets needed
ctest --test-dir build -C Release               # full parity: needs ggufs/ + dumps/
```

## Build

```sh
git clone --recursive <this-repo> trellis2cpp
cd trellis2cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On Windows the configure scripts in the repository root select the backend for
you (see the quick starts). If you cloned without `--recursive`:

```sh
git submodule update --init --recursive
```

CGAL is auto-detected. Install CGAL 5.5 or newer before configuring to enable
the portable CPU print-remesh backend, or pass `-DTRELLIS2_CGAL=OFF` to disable
the probe. CMake prints whether Alpha Wrap was enabled. The standard demo
container already includes CGAL 6.1.1.

## Repository layout

The annotated tree, with what each directory is for, is in
[`docs/architecture/README.md`](docs/architecture/README.md#directory-structure)
and [`AGENTS.md`](AGENTS.md#directory-structure). The short version: `src/` is
the library and its C ABI, `examples/` the per-stage CLIs, `tests/` the parity
gates, `server/` the Go demo and its embedded viewer, `scripts/` the converters
and reference dumps, `third_party/` vendored MIT dependencies, `ggml/` the
submodule, and `docs/` everything written down.

## License

MIT. See [LICENSE](LICENSE). Every third-party component, its license, what it
is used for and how it is obtained is listed in
[THIRD_PARTY.md](THIRD_PARTY.md) — read that before adding a dependency.

Upstream is MIT too, and the copyright of the base pipeline stays with its
authors: `LICENSE` names Ryan Schmidt (rms80) for the stage-1 geometry port,
Richard Palethorpe for the image→mesh pipeline, and Robert Beckebans for this
fork. Nothing in this repository is adopted
from the TRELLIS.2 reference implementation itself: its behaviour and file
formats are reimplemented under our own design, which is why a checkout of the
reference is kept out of the tree even though it is MIT as well.

Vendored code under `third_party/` is also MIT:
[meshoptimizer](https://github.com/zeux/meshoptimizer) (Arseny Kapoulkine),
[xatlas](https://github.com/jpcy/xatlas) (Jonathan Young),
[tinybvh](https://github.com/jbikker/tinybvh) (Jacco Bikker) and the
[AutoRemesher](https://github.com/huxingyi/autoremesher) core with
isotropicremesher (Dust3D Project / Jeremy HU), plus
[stb](https://github.com/nothings/stb) (public domain / MIT).

Two optional dependencies are resolved at configure time rather than vendored:

- The Alpha Wrap backend links against [CGAL](https://www.cgal.org/) 5.5 or
  newer. CGAL's 3D Alpha Wrapping package is GPL-3.0-or-later (or available
  under a commercial CGAL license), so binaries built with `TRELLIS2_CGAL=ON`
  **and** CGAL detected are subject to those terms. It is a vcpkg manifest
  *feature*, so a default configure cannot pull it in by accident.
- The quad remesh stage needs [Eigen](https://eigen.tuxfamily.org/) 5.x, which
  is MPL-2.0. It is used unmodified and never vendored, so no MPL2 file enters
  this repository; MPL-2.0 is file-level copyleft and explicitly permits
  combination with a differently licensed Larger Work.
