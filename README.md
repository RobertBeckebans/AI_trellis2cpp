# trellis2.cpp

A C++/[ggml](https://github.com/ggml-org/ggml) implementation of the
**TRELLIS.2** image-to-3D pipeline: an image goes in, a 3D mesh with per-vertex
PBR textures comes out, with all inference in C++/ggml (no PyTorch at runtime).
The demo can also export the result into a portable, full-density **GLB** with
standard interpolated vertex colour and retained PBR attributes—no reference
container required.

Modeled structurally after [sam3.cpp](https://github.com/rms80/sam3.cpp):
single-file library (`src/trellis2.h` / `src/trellis2.cpp`), bundled ggml as a
submodule (Metal on by default on Apple), DLL-export decoration, and a
CMake build with example executables. A flat C ABI (`src/trellis2_capi.h`) drives
a Go demo server with a browser mesh viewer.

## What this fork adds

This is a **downstream fork**. The port itself is upstream work: the stage-1
geometry port by [rms80](https://github.com/rms80), and the complete
image→mesh pipeline — DINOv3 encoder, shape-SLAT stage, demo server, PBR
texturing, CGAL print wrap — by
[Richard Palethorpe](https://github.com/richiejp), through
`84f4a2c`. Everything below was added on top of that base; check `git log`
before attributing a design decision.

**Runs on AMD, and on Windows.** Upstream targets Linux and CUDA through a
container. This fork adds a HIP/ROCm backend (developed on a Radeon AI PRO
R9700, `gfx1201`), a Vulkan backend, a Windows DLL build with the Go server
alongside it, and ready-made configure scripts for all three. The CUDA path is
kept and still builds, but it is not exercised here for want of the hardware,
and it does inherit one change: ggml's CUDA graphs are disabled by default,
because they crash on HIP and the two share that code. `TRELLIS2_CUDA_GRAPHS=1`
turns them back on.

**Both GPU backends live in one build and switch at runtime.** ggml registers
each with the same device registry, so `TRELLIS2_DEVICE=cpu|rocm|vulkan` — or a
dropdown in the viewer — picks between them without a second build tree or a
restart. The backend a mesh was produced on is recorded in its generation
manifest, because two meshes from one seed otherwise differ for a reason
nothing preserves.

**1536³ cascade tier**, with a token budget that steps the resolution back down
when the scaffold would exceed it, rather than failing at the decode.

**Quad retopology and a normal-map bake**, so generated geometry is usable
downstream: the [AutoRemesher](https://github.com/huxingyi/autoremesher) core
vendored and built without Qt or TBB, reachable from the C ABI, the server and
the viewer; plus a tangent-space normal map baked into the exported GLB so
replacement geometry keeps the original detail. In practice this is the
`print+quad` combination — Alpha Wrap hands the remesher the closed 2-manifold
it needs, and the bake carries the dense mesh's detail onto the result. Read
the licence note below before assuming it is reachable in a default build. The bake is in **MikkTSpace**,
using [meshoptimizer](https://github.com/zeux/meshoptimizer)'s
MikkTSpace-compatible generator (`meshopt_generateTangents` with
`meshopt_TangentCompatible`). That is not a concession to one tool: MikkTSpace
is what glTF 2.0 prescribes — a consumer that finds no `TANGENT` attribute is
told to derive tangents with the standard MikkTSpace algorithm — so it is the
only basis a normal map can be baked against and stay correct across
conformant importers. Blender's, which recomputes rather than reading the
attribute, is simply where a mismatched basis becomes visible first.

**The GPL surface is smaller, but the good retopology still needs it.** The
closest-surface PBR projection moved from CGAL to
[tinybvh](https://github.com/jbikker/tinybvh), so a default configure pulls in
no copyleft and produces generations and full-density GLB exports on MIT terms
alone. That is not the same as the whole feature set being license-free: the
usable low-poly path still runs through CGAL's Alpha Wrap, because AutoRemesher
wants a closed 2-manifold and does not cope with raw dual-grid geometry at
these resolutions. Without Alpha Wrap there is no low-poly target for the bake
to project onto, which makes the tinybvh projection moot in that
configuration. See D4 in
[`docs/plan/autoremesher-quad-remesh.md`](docs/plan/autoremesher-quad-remesh.md)
for the mode matrix and which combination is which license.

**A silent GPU defect found, isolated, and worked around by chunking.** On
`gfx1201`, `mul_mat` stops writing past a fixed column count and still reports
success — which sheared roughly a quarter off every 1024³ mesh with nothing
logged anywhere. It is not ggml and not the hardware: reproduced in ten lines
of plain PyTorch, absent on Vulkan on the same card, and independently hit by
the ROCm fork of TRELLIS.2 itself, which chunks its own GEMMs for the same
reason. The fix here is the same shape — the decoder evaluates its finest level
in voxel blocks so no single matmul reaches the ceiling. That is
bit-identical to the unchunked path rather than merely close
(`tests/test_chunked_decode`, `max|d| = 0`), and costs nothing measurable in
wall clock. `TRELLIS2_NO_CHUNK=1` turns it off to reproduce the artefact
deliberately. Written up in
[`docs/bugs/ggml-rocm-mul-mat-column-limit.md`](docs/bugs/ggml-rocm-mul-mat-column-limit.md)
with a standalone reproducer.

**Numerical parity you can actually run.** The PyTorch reference environment is
reproducible natively through `uv`, including AMD's Windows ROCm wheels, so the
golden dumps no longer need the CUDA container. The reference implementation
can also produce a *whole generation* that lands in the viewer next to the
backend runs of the same image
([`scripts/ref_generate.py`](scripts/ref_generate.py)), which is what turned
"the backends look different" into numbers. Tap-by-tap status is in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

**Diagnosability throughout**: per-stage timings and a progress trace for
server work, a report of which projection backend a build uses, and warnings
when the decoder's subdivision runs away *or* collapses — the latter added
because the worst failure on record passed the one-sided check in silence.

## Quick start

Windows with an AMD card — the primary path for this fork. No container.

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
- `cmake-ninja-win64-rocm.bat` needs the **ROCm SDK** and the **Vulkan SDK**
  (it refuses to start without `VULKAN_SDK`, since it builds both backends into
  one library). To skip Vulkan, set `-DGGML_VULKAN=OFF` in the script, or use
  `cmake-ninja-win64-cpu.bat` / `cmake-ninja-win64-vulkan.bat` instead. It
  **wipes `build/`**, so it is a configure step, not a rebuild step.
- `start_server.bat` is the normal way to start: it rebuilds whatever is stale
  and puts the DLL and the ROCm runtime on `PATH` before launching.

The device selector in the viewer then switches between CPU, ROCm and Vulkan
without a restart.

## Quick start (docker, CUDA)

The upstream container path, unchanged. Linux and an NVIDIA card.

```sh
git submodule update --init --depth 1                 # ggml
scripts/download_ggufs.sh                             # prebuilt f16 GGUFs -> ggufs/ (~14 GB)
docker build -f docker/Dockerfile.demo -t trellis2-demo docker   # CUDA runtime + Go
# build the CUDA shared lib + Go server, then run
docker run --rm -v "$PWD":/work -w /work trellis2-demo bash -c '
  cmake -B build-cuda-shared -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SHARED_LIBS=ON && cmake --build build-cuda-shared -j
  cd server && go build -o trellis2-server-linux .'
docker run --rm --device nvidia.com/gpu=all -v "$PWD":/work -w /work/server -p 8742:8742 \
  trellis2-demo ./trellis2-server-linux -lib /work/build-cuda-shared/libtrellis2.so \
  -ggufs /work/ggufs -store /work/generations -unload-idle
# open http://localhost:8742 and drop an image
```

Or just run `scripts/demo.sh`, which builds the lib + server, auto-downloads any
missing GGUFs, and launches the container.

## Prebuilt GGUFs

Applies to both quick starts. `scripts/download_ggufs.sh` pulls the ready-made
f16 GGUFs from three public repos
under the [LocalAI-io](https://huggingface.co/LocalAI-io) org, so you can skip the
safetensors download and the conversion step entirely:

- [`TRELLIS.2-4B-GGUF`](https://huggingface.co/LocalAI-io/TRELLIS.2-4B-GGUF) — MIT
- [`TRELLIS-image-large-GGUF`](https://huggingface.co/LocalAI-io/TRELLIS-image-large-GGUF) — MIT
- [`dinov3-vitl16-pretrain-lvd1689m-GGUF`](https://huggingface.co/LocalAI-io/dinov3-vitl16-pretrain-lvd1689m-GGUF) — DINOv3 License (Built with DINOv3)

**Developers** who need the f32 validation variants, or who want to regenerate the
GGUFs from source, use the original flow instead: `scripts/download_models.sh` (HF
safetensors → `models/`, ~7 GB) then `docker run … trellis2-ref bash
scripts/convert_all.sh` (safetensors → GGUF, f16 + f32). The card + license sources
for the published repos live in `scripts/hf/`, and `scripts/upload_ggufs.sh`
(re)publishes them. The conversion also runs without the container — see
[`docs/reference-environment.md`](docs/reference-environment.md), which is the
same `uv` environment the parity dumps use.

## The demo server

Completed generations are committed atomically under `generations/` (final
mesh, replay frames, and manifest) and restored with the same job IDs after a
server restart. Pass `-store ''` to disable persistence or `-store PATH` to use
a different durable location. Incomplete writes are ignored on startup. With
`-unload-idle`, the HTTP server starts without allocating model VRAM, loads the
pipeline on the first generation, and releases it again when the queue is idle.

The browser UI has a **quality** selector: coarse preview (64³ marching cubes),
512³ fine, **1024³ cascade** (the TRELLIS.2 default), or **1536³ cascade**. The
1536 tier reuses the 1024 checkpoints and applies a token budget that steps the
resolution back down when the scaffold would exceed it, so a request that will
not fit degrades instead of failing at the decode. Coarse falls
back automatically if the shape-SLAT models are absent (`-coarse`); both
cascades need the extra 1024 model (`-no-1024` disables them). A **compute
device** selector sits above quality: one build carries CPU, ROCm and Vulkan,
and switching drops the resident models so the next generation reloads on the
chosen backend.
Enable **free VRAM when idle** to unload the resident model pipeline between
generations; the next queued generation reloads it automatically.
**Live steps** is off by default because each sparse-structure frame requires an
extra CPU occupancy decode between GPU inference steps. Its button always says
`on` or `off`; enabling it records the frames used by replay and showcase mode.
Completed jobs expose `durationMs`, `livePreview`, and per-stage `stageTimings`
through `/api/job/{id}` and persist those diagnostics in their manifest.
The always-visible **asset export** panel preserves the generated polygon count,
can preview component cleanup, optionally keep only the largest connected piece,
restore the original preview, and download a Three.js-ready GLB. Dense generated
materials are stored as standard interpolated vertex colours rather than a
sub-texel per-triangle atlas; original metallic/roughness values are also retained
in the custom `_METALLIC_ROUGHNESS` attribute. All components are preserved by
default; destructive cleanup must be selected explicitly. Showcase mode likewise loads the original
full-density mesh. Open `/showcase` for the separate full-screen storyboard: it
starts each generation with its saved source image centered, moves that image to
the upper-right, replays the recorded stages, and then lingers on a slowly
rotating final model. New generations retain the original upload byte-for-byte
for display, plus the full-resolution processed PNG actually used by TRELLIS for
repeatable server-side regeneration without another upload. Select a saved mesh
and use **regenerate from saved image** to run it again with the current settings.
Older manifests fall back to their thumbnail and can only regenerate when their
older processed source file is available.

When built with CGAL 5.5 or newer, asset export also offers **watertight print
wrap**. It runs CPU Alpha Wrap over the selected components and previews the
exact replacement geometry before download. CGAL guarantees the result is
closed, oriented, intersection-free, and 2-manifold. `detail size` controls the
smallest holes/cavities the wrap enters; `offset` controls how tightly it encloses
the generated surface. Both are percentages of the source bounding-box diagonal.
Because wrapping creates new vertices, its fast browser preview is geometry-only.
When the source is textured, GLB download automatically unwraps the print mesh
with xatlas and rebakes base color, metallic, roughness, and opacity: each atlas
texel is projected through a CGAL AABB tree to the closest source triangle and
receives its barycentrically interpolated dense PBR. This mirrors upstream's
GPU remesh texture-transfer strategy on the CPU. Sources without PBR remain grey.
This fixes solid topology; physical scale, minimum wall thickness, and supports
still need to be chosen for the target printer/material.
The same path is available without the server:

```sh
./build/examples/mesh2glb mesh.t2mesh printable.glb --print 1 0.0333
# percentages: alpha/detail size, then enclosing offset
```

On a 16 GB RTX 50-series: the 512 fine path runs image→mesh in ~110 s (~1M-vertex
512³ mesh); the 1024 cascade adds a second 1.3B-model pass and the 1024³ decoder
for a ~5M-vertex mesh (~5 min, ~10 GB VRAM, and a ~14 GB host-RAM spike for the
1024³ sparse-conv decode).

## Pipeline

```
image (RGB/RGBA)
  → background cleanup  border-connected black/white → feathered alpha        [C++/browser]
  → preprocess          alpha bbox crop, premultiply, PIL-exact Lanczos-512   [C++, byte-exact]
  → DINOv3 ViT-L/16     [1, 1029, 1024] conditioning tokens                   [C++/ggml]
  → SS-flow DiT         1.3B dense DiT, 12-step CFG flow-Euler → z_s          [C++/ggml]
  → SS decoder          dense 3D-conv → 64³ occupancy → 32³ voxel scaffold    [C++/ggml]
  → shape-SLAT DiT      1.3B sparse DiT over active voxels, 12-step CFG       [C++/ggml]
  → shape VAE decoder   sparse ConvNeXt U-Net, 16× up → decoded dual grid      [C++/ggml]
  ├→ flexible dual grid → triangle mesh                                        [C++]
  └→ shape VAE encoder  validated dual-grid → shape SLat + subdivision guide   [C++/ggml]
     → texture-SLAT DiT shape-SLat concat conditioning                         [C++/ggml]
     → texture decoder  replay subdivision → sparse 6-channel PBR volume       [C++/ggml]
  → material sampling   trilinear PBR at surface vertices                      [C++]
```

The **1024 cascade** (default in TRELLIS.2) adds a second pass on top: the 512
result's decoder `.upsample(×4)` predicts a denser coordinate scaffold, which is
quantized to 64³ and fed to a second 1.3B shape-SLAT flow (the 1024 model,
conditioned on a 1024-res DINOv3 encode) and the same decoder at 1024³ — a
~5M-vertex mesh. The ~49k-token HR attention only fits in VRAM via flash
attention (`sdpa_auto`); see [docs/VERIFICATION.md](docs/VERIFICATION.md).

The neural components are validated tap-by-tap against the PyTorch reference,
with separate integration regressions for subdivision guidance, sparse material
sampling, and GLB alpha preservation — see
[docs/VERIFICATION.md](docs/VERIFICATION.md). Highlights: preprocessing is
byte-exact, the DINOv3 encoder matches to rel-L2 ≤ 7e-7 across 40 taps, and the
sparse U-Net decoder is numerically exact through all four conv levels.

## Components

- **Image preprocessing + DINOv3 encoder** — `trellis2_preprocess_rgba()`
  reproduces `pipeline.preprocess_image` (the has-alpha path) with a
  PIL-compatible fixed-point Lanczos-3 resampler (byte-exact vs Pillow).
  `trellis2_remove_solid_background_rgba()` first converts a detected near-black
  or near-white background connected to the image border into softly feathered
  alpha, while preserving enclosed black/white subject details and existing
  alpha masks. The demo exposes automatic, forced-black, forced-white, and keep
  original modes.
  `trellis2_dino_encode()` runs the full DINOv3 ViT-L/16 (axial-2D RoPE,
  LayerScale, exact-GELU MLP) and applies the affine-free final LayerNorm the
  flow models expect — the `[1, 1029, 1024]` conditioning that used to come
  from an external `dump_dinodata.py`. `dino_encode` chains them:

  ```sh
  ./build/examples/dino_encode ggufs/dino_f16.gguf image.png cond.dinodata
  ```

- **`.dinodata` loader** — `trellis2_load_dinodata()` still reads/writes the
  precomputed conditioning tensor (1 CLS + 4 register + 1024 patch, last layer,
  affine-free LN; `neg_cond = zeros_like(cond)`), for testing and CLI chaining.

- **SS-flow DiT weights** — `convert_ss_flow_to_gguf.py` converts the stage-1
  `ss_flow_img_dit_1_3B_64_bf16` checkpoint to GGUF; `trellis2_ss_flow_load()`
  reads it back through ggml (hparams from `trellis2.ss_flow.*` KV metadata,
  weights keyed by their original checkpoint names).

- **SS-flow DiT forward pass** — `trellis2_ss_flow_forward()` builds the full
  ggml graph: input projection, sinusoidal timestep + shared adaLN-Zero
  modulation, 30 cross-blocks (self-attention with 3D interleaved RoPE +
  QK-RMSNorm, cross-attention to the DINOv3 tokens, GELU-tanh FFN), and the
  final LayerNorm + output projection. Runs on an **auto-selected backend** —
  the first GPU exposed by ggml (CUDA / Metal / Vulkan / ...), falling back to
  CPU, like sam3.cpp. Validated against a PyTorch f32 reference to **<1e-3
  relative L2** on CPU, Metal (f32), and Metal (f16) (see *Validation* below).

- **Stage-1 sampler** — `trellis2_ss_flow_sample()` runs the full flow-Euler
  loop with classifier-free guidance (interval [0.6,1.0], strength 7.5, rescale
  0.7, rescale_t 5.0, 12 steps; `neg_cond = zeros`) to turn a DINOv3 cond into
  the sparse-structure latent z_s. Validated against the real
  `FlowEulerGuidanceIntervalSampler`: **rel L2 5.7e-3, 99.85% sign agreement**
  (the SS decoder thresholds z_s at 0). Run it:

  ```sh
  ./build/examples/ss_sample ss_flow_dit_f16.gguf /path/img.dinodata out.latent
  # -> z_s [8,16,16,16], occupancy(>0) ~50%
  ```

- **Stage-1 SS decoder** — `trellis2_ss_dec_decode()` runs the
  `SparseStructureDecoder` (a dense 3D-conv ResNet) that turns the z_s latent
  `[8,16³]` into an occupancy logit grid `[1,64³]`, upsampling 16→32→64 with two
  `pixel_shuffle_3d` blocks. The coarse voxel scaffold is `logit > 0`. Runs fully
  on the GPU (ggml `conv_3d_direct`, channel-LayerNorm, in-graph pixel-shuffle).
  Validated against the real PyTorch decoder to **rel L2 5e-7 (f32) / 2e-5 (f16),
  100% sign agreement** on a sampled z_s. Run it:

  ```sh
  ./build/examples/ss_decode ss_dec_f16.gguf out.latent out.occ
  # -> logits [1,64,64,64], occupied(>0) grid (the coarse voxel scaffold)
  ```

- **Occupancy → coarse mesh** — `ss_mesh` decodes a z_s latent and exports the
  `{logit = 0}` isosurface as a watertight OBJ via a self-contained marching
  cubes (`examples/marching_cubes.h`, the tetrahedral / Freudenthal variant — no
  256-row table, provably manifold). This is the fast preview path:

  ```sh
  ./build/examples/ss_sample ss_flow_dit_f16.gguf /path/img.dinodata z_s.latent
  ./build/examples/ss_mesh   ss_dec_f16.gguf z_s.latent shape.obj --normalize
  # -> watertight shape.obj in the centered unit cube; open in any 3D viewer
  ```

- **Shape-SLAT flow + decoder (fine geometry)** — `trellis2_slat_flow_sample()`
  runs the sparse 1.3B DiT over the active voxels of the 32³ scaffold (same
  block structure as the SS-flow DiT, 3D RoPE over each voxel's coords),
  denormalized with `shape_slat_normalization` baked into the GGUF.
  `trellis2_shape_dec_decode()` runs `FlexiDualGridVaeDecoder` — a sparse
  ConvNeXt U-Net whose 3×3×3 submanifold convolutions are expressed as 27
  gather+GEMM steps, with each level's learned subdivision growing the active
  set (32³ → 512³, 16×). `examples/flexible_dual_grid.h` turns the 7-channel
  per-voxel output (dual-vertex offset, per-axis intersection flags, quad split
  weight) into the triangle mesh. This is the real TRELLIS.2 geometry, driven
  end-to-end by the demo server.

- **PBR texture generation** — the decoded dual grid is encoded to the shape
  SLat used to condition texture flow, using the numerically validated
  standalone texturing path. The decoded six-channel volume (base color,
  metallic, roughness, alpha) is sampled trilinearly at the actual dual-grid
  surface positions. Collapsed all-saturated outputs are rejected instead of
  being persisted as apparently successful textures. The browser linearizes base
  color before PBR lighting and preserves opacity. Material sampling steps are
  controlled separately from geometry steps, matching upstream's defaults.

## Validate the forward pass

```sh
# 1. lossless f32 weights for an exact comparison
python convert_ss_flow_to_gguf.py --output ss_flow_dit_f32.gguf --ftype 0

# 2. PyTorch f32 reference forward -> tests/ss_flow_ref.bin
python tests/ref_ss_flow.py --dinodata /path/MushroomBoy.dinodata

# 3. build + run the C++ comparison
cmake -B build -DTRELLIS2_BUILD_TESTS=ON && cmake --build build -j
./build/tests/test_ss_flow_forward ss_flow_dit_f32.gguf tests/ss_flow_ref.bin
# -> rel L2 err ~2.8e-4, RESULT: PASS
```

## Validate the SS decoder

```sh
# 1. lossless f32 decoder weights
python convert_ss_dec_to_gguf.py --output ss_dec_f32.gguf --ftype 0

# 2. PyTorch f32 reference decode of a sampled z_s -> tests/ss_dec_ref.bin
./build/examples/ss_sample ss_flow_dit_f16.gguf /path/img.dinodata z_s.latent
python tests/ref_ss_dec.py --latent z_s.latent

# 3. build + run the C++ comparison
./build/tests/test_ss_dec ss_dec_f32.gguf tests/ss_dec_ref.bin
# -> rel L2 err ~5e-7, RESULT: PASS
```

## Convert the stage-1 weights

```sh
# needs safetensors + torch + numpy (e.g. the trellis2-shiv venv)
python convert_ss_flow_to_gguf.py --output ss_flow_dit_f16.gguf --ftype 1   # DiT
python convert_ss_dec_to_gguf.py  --output ss_dec_f16.gguf      --ftype 1   # decoder
```

`--model` / `--config` default to the `microsoft/TRELLIS.2-4B` HF cache
snapshot. `--ftype`: `0` = f32 (lossless upcast from bf16), `1` = f16
(default — big 2-D weight matrices only; norms/gammas/modulation stay f32),
`2` = bf16 (lossless, needs bf16-capable ggml). The f16 file is ~2.6 GB.

Inspect it (validates that ggml can read every tensor):

```sh
./build/examples/ss_flow_info ss_flow_dit_f16.gguf          # metadata only
./build/examples/ss_flow_info ss_flow_dit_f16.gguf --load   # + read all weights
```

## Build

```sh
git clone --recursive <this-repo> trellis2cpp
cd trellis2cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On Windows the configure scripts in the repository root do this with the right
backend already selected — `cmake-ninja-win64-rocm.bat` (HIP **and** Vulkan in
one library, the primary build for this fork), `cmake-ninja-win64-vulkan.bat`,
or `cmake-ninja-win64-cpu.bat`. They wipe `build/` first. `start_server.bat`
then builds what is stale and launches the Go server with the DLL and the ROCm
runtime on `PATH`; that `PATH` also matters for `ctest`, which otherwise blocks
in the loader rather than failing.

CGAL is auto-detected. Install CGAL 5.5 or newer before configuring to enable
the portable CPU print-remesh backend, or pass `-DTRELLIS2_CGAL=OFF` to disable
the probe explicitly. CMake prints whether Alpha Wrap was enabled. The standard
demo container already includes CGAL 6.1.1.

If you already cloned without `--recursive`:

```sh
git submodule update --init --recursive
```

## Try it

```sh
./build/examples/dino_info /path/to/MushroomBoy.dinodata
```

Prints the shape, token breakdown, and fingerprints (min/max/mean/sum/l2).
`min`/`max`/`count` match the matching `<stem>.dino.txt` JSON sidecar exactly
(they are true element values); `sum`/`l2` agree to float32 precision — the C++
side reduces in `double` and is slightly more accurate than numpy's float32
reduction.

## Layout

| path           | what                                                   |
|----------------|--------------------------------------------------------|
| `src/trellis2.h` | public API (DLL-decorated, versioned)                  |
| `trellis2.cpp` | implementation                                         |
| `convert_ss_flow_to_gguf.py` | stage-1 DiT checkpoint → GGUF converter  |
| `convert_ss_dec_to_gguf.py`  | stage-1 decoder checkpoint → GGUF converter |
| `src/mesh_export.{h,cpp}` | CUDA-free GLB export with direct vertex PBR or projected UV-atlas textures |
| `src/print_remesh.{h,cpp}` | optional CGAL Alpha Wrap reconstruction and closest-surface PBR transfer |
| `examples/`    | CLI tools (`dino_info`, `ss_flow_info`, `ss_sample`, `ss_decode`, `ss_mesh`, `mesh2glb`) |
| `examples/marching_cubes.h` | single-file isosurface → OBJ extractor      |
| `src/quad_remesh.{h,cpp}` | optional AutoRemesher quad-dominant mid-poly retopology |
| `third_party/` | vendored `xatlas` (print-wrap and opt-in ordinary UV unwrap), `meshoptimizer`, `tinybvh`, the `autoremesher` core, and `stb` for image I/O |
| `ggml/`        | submodule, tracking upstream ggml                      |

## License

MIT. See [LICENSE](LICENSE). Every third-party component, its license, what it
is used for and how it is obtained is listed in
[THIRD_PARTY.md](THIRD_PARTY.md) — read that before adding a dependency.

Upstream is MIT too, and the copyright of the base pipeline stays with its
authors — rms80 and Richard Palethorpe. Nothing in this repository is adopted
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
