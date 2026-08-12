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

## Quick start (demo)

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

### Prebuilt GGUFs

`scripts/download_ggufs.sh` pulls the ready-made f16 GGUFs from three public repos
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
(re)publishes them.

Completed generations are committed atomically under `generations/` (final
mesh, replay frames, and manifest) and restored with the same job IDs after a
server restart. Pass `-store ''` to disable persistence or `-store PATH` to use
a different durable location. Incomplete writes are ignored on startup. With
`-unload-idle`, the HTTP server starts without allocating model VRAM, loads the
pipeline on the first generation, and releases it again when the queue is idle.

The browser UI has a **quality** selector: coarse preview (64³ marching cubes),
512³ fine, or **1024³ cascade** (the TRELLIS.2 default and highest resolution
currently supported here). Upstream's optional 1536 cascade is not yet ported. Coarse falls
back automatically if the shape-SLAT models are absent (`-coarse`); the 1024
cascade needs the extra 1024 model (`-no-1024` disables it).
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
