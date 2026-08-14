# trellis2.cpp completion plan

Goal: image in → 3D mesh out, pure C++/ggml inference behind a Go demo server with a
browser mesh viewer. No PBR textures for now. Process mirrors depth-anything.cpp
(layer-by-layer parity vs the PyTorch reference) and privacy-filter.cpp (libFuzzer
harnesses on untrusted inputs, sanitizer builds).

## Pipeline target

TRELLIS.2 "512" pipeline type (non-cascade), geometry only:

```
image (RGBA preferred)
  → preprocess (alpha crop, premultiply, resize 512, ImageNet norm)   [C++ port]
  → DINOv3 ViT-L/16 cond tokens [1, 1+4+1024, 1024]                   [C++ port — was external .dinodata]
  → SS-flow DiT (1.3B dense, 12-step CFG flow Euler)                  [already ported + validated]
  → SS decoder → 64³ occupancy → max_pool 32³ → coords                [already ported; pool/coords new]
  → shape-SLAT flow (1.3B sparse DiT, res 32, 32ch)                   [to port]
  → FlexiDualGrid VAE decoder (sparse ConvNeXt U-Net, 16× up)         [to port]
  → flexible dual grid → triangle mesh                                [to port, CPU]
  → (postprocess: fill holes — best effort, CuMesh not portable)
```

Fallback shipped at every point in time: marching-cubes preview mesh from the 64³
occupancy (already works), so the demo is demoable before the SLAT stages land.

The upstream neural background-removal net (BiRefNet/RMBG-2.0, a separate ~1 GB
model) remains out of scope. The port does include deterministic removal of
near-black/near-white backgrounds connected to the image border, with soft alpha
edges and browser controls to force or disable it. Texture and cascade status is
tracked below.

## Weights

`scripts/download_models.sh` → `models/`. DINOv3 comes from the ungated
`camenduru/dinov3-vitl16-pretrain-lvd1689m` mirror because the official
`facebook/dinov3-vitl16-pretrain-lvd1689m` is license-gated and this account has no
access yet (403). Request access and re-download officially when possible.

## Validation (depth-anything.cpp process)

- Reference activations dumped from Python via forward hooks into
  `dumps/reference_<stage>.gguf` + manifest (GGUF as the dump format so C++ reads it
  with the ggml API it already links).
- Reference env: docker `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` + stock pip
  deps only. Custom CUDA backends (FlexGEMM/flash-attn/o-voxel) are NOT installed;
  sparse attention (batch=1) is monkeypatched to dense SDPA and sparse conv to a
  pure-torch gather-GEMM — slow but runs on CPU too, and doubles as an executable
  spec for the C++ port. 16 GB VRAM is only a constraint for the reference runs;
  stage-at-a-time + low_vram-style model swapping keeps us under it.
- C++ side: per-layer taps compared with `tests/parity.hpp`-style
  `atol+rtol*|ref|` gate (2e-3 default), ctest with `SKIP_RETURN_CODE 77` when
  fixtures are absent.
- Existing whole-stage tests (ss_flow/ss_sample/ss_dec) keep working; their
  ref generators get path fixes (the `trellis2-shiv` layout doesn't exist here).

## Fuzzing (privacy-filter.cpp process)

Non-GGUF untrusted inputs, libFuzzer + ASan/UBSan (clang):
- image bytes → stb_image decode → preprocess (the demo upload path)
- `.dinodata` loader
- `.latent`/occupancy readers used by the example CLIs
GGUF model files are trusted assets (same threat model as privacy-filter).

## Demo

- `server/` Go (stdlib http + purego dlopen of `libtrellis2.so`), depth-anything
  server pattern: single inference mutex, `POST /api/generate` (multipart image,
  params) → job id, progress polling, `GET /api/mesh/<id>` binary mesh,
  self-contained embedded `web/index.html` WebGL viewer (no CDN, no build step).
- C API additions in `trellis2.h` for: dino encode from RGBA buffer, full pipeline
  run with progress callback, mesh buffer accessors.

## Order of work

1. infra: ref container, weights, fix existing ref scripts, regenerate refs,
   existing 3 tests green on this machine (CPU + asan)
2. DINOv3 encoder port + per-layer validation → image→coarse-mesh e2e in C++
3. Go server + web viewer on coarse pipeline (user-visible milestone)
4. fuzz harnesses + short campaigns
5. shape-SLAT flow port (sparse DiT) + validation
6. FDG VAE decoder + flexible-dual-grid mesher + validation → real mesh in demo
7. docs (VERIFICATION.md), CUDA build check, quantized variants, benchmarks

## Status (done)

All of 1–7 complete. Image→mesh works end-to-end, C++/ggml only, coarse
(marching-cubes preview) and fine (512³ dual-grid) paths, in the Go demo. Every
stage validated tap-by-tap (docs/VERIFICATION.md): preprocessing byte-exact,
DINOv3 rel-L2 ≤7e-7, sparse U-Net decoder exact through 4 levels. CUDA build
(sm_120) works; 3D-conv decoders pinned to CPU. One fuzz bug found+fixed.

Runtime on the 16 GB RTX 5070 Ti (GPU shape decode auto-enabled): fine (512³)
**~39 s**, 1024 cascade **~133 s**. (Was 85 s for 512 with the decoder on CPU.)

### Where the fine-path time goes (measured, `TRELLIS2_TIMING=1`)

Per-stage wall-clock for one 512³ generation (f16, ~1.09 M-vertex mesh), CPU
decoder vs the auto GPU decoder:

| stage      | CPU decode | GPU decode | device (GPU mode) |
|------------|-----------:|-----------:|-------------------|
| dino       | 0.1 s      | 0.1 s      | GPU               |
| ss_flow    | 21 s       | 21 s       | GPU               |
| ss_dec     | 3 s        | 3 s        | CPU (dense CONV_3D)|
| slat_flow  | 11 s       | 10 s       | GPU               |
| shape_dec  | **49 s**   | **2.5 s**  | **GPU** (was 59 % of the run) |
| mesh       | 0.2 s      | 0.2 s      | CPU               |
| **total**  | **85 s**   | **39 s**   |                   |

The original "GPU ~30 %" reading was the *average*: the biggest stage (the
FlexiDualGrid VAE decoder) ran on the CPU with the GPU idle. It now runs on the
GPU — **~2.5 s vs ~49 s, ~20×** — placed there automatically when VRAM allows
(see the shape-decoder note under "Not done"). The flow DiTs (`ss_flow` +
`slat_flow`) are now the bulk of the run. Two facts about them, from
`TRELLIS2_TIMING`:

- Attention is now `ggml_flash_attn_ext` for **all** flow forwards, not just the
  cascade's huge token counts (`sdpa_auto`, `TRELLIS2_SDPA_EXACT` restores the
  old materialized softmax). Flash is bit-identical to full softmax on the CPU
  (both flow forwards still validate at ~2.4e-4 / 2.9e-4 rel-L2) and on the GPU
  it is ~30 % faster *and* O(L) memory — the exact path's 805 MB score matrix
  `alloc_graph`-fails once the resident pipeline leaves <~2 GB free, so flash is
  what keeps the flow fitting under host-VRAM pressure, not only a speedup.
- Each forward still runs at only ~17 % of the card's f16 tensor-core peak; the
  matmuls already hit the f16 cores (ggml converts the f32 activations), so the
  remaining slack is the f32 elementwise/permute traffic between matmuls. CUDA
  graphs do not help (they `cudaGraphInstantiate`-OOM with the pipeline
  resident) and are left off.

The shape decoder is a *sparse* net that keeps millions of voxels in the tensor
**row** dimension (`ne1`). ggml's CUDA CONCAT/PAD kernels launch a grid
dimension equal to that count and abort (`invalid configuration argument`) above
the 65535 grid-Y/Z limit — which is why the original port used the CPU. But
CONCAT/PAD were only used to append one missing-neighbor zero row; replacing
that with a clamp-index + 0/1-mask formulation (get_rows a valid row, multiply
the missing ones by zero — byte-identical) keeps every op inside ggml kernels
that tile the voxel dimension (`get_rows`, broadcast `mul`, `mul_mat`, `add`,
`norm`, `silu` all verified OK at 3 M voxels). The decoder then runs on the GPU
in ~2.5 s (512³) / ~12.5 s (1024³).

### VRAM-aware auto-placement of the shape decoder

The decoder's placement is decided from **measured free VRAM**
(`trellis2_gpu_free_vram()` → `cudaMemGetInfo`), not a hardcoded flag:

- At load, `t2_pipeline_load` records the free VRAM *before* the flow DiTs are
  loaded (= what freeing them again reclaims) and places the decoder on the GPU
  when that covers the target tier's decode peak (~3 GB at 512³, ~7.5 GB at
  1024³) plus the decoder weights and a margin; otherwise CPU. `TRELLIS2_SHAPE_DEC_GPU`
  / `_CPU` force it; a CPU-only build (no GPU device) always picks CPU.
- At decode time, `ensure_decode_vram` checks free VRAM again and, if the decode
  would not fit, frees the flow DiTs (all finished by then) to reclaim their
  ~5–7 GB; `reload_flows` brings them back on the next `t2_generate` (a few
  seconds from page cache). So 512³ decodes with the flows resident (no reload
  cost) while the big 1024³ decode transparently frees-and-reloads. This is the
  `T2_LOAD_LOW_VRAM` idea, applied automatically only when the numbers require it.

Validated end-to-end on the 16 GB card with **no env vars**: 512 → 39 s, 1024
cascade → 133 s (the 1024³ GPU decode is 12.5 s), and three back-to-back cascade
generations exercise the free/reload path with no OOM.

## Not done (out of original scope / future)

- **PBR textures — DONE.** The validated standalone texturing path re-encodes
  the decoded dual grid to condition texture flow, then trilinearly samples the
  decoded six-channel PBR volume at the surface. The browser and GLB path
  preserve alpha and use the correct base-color space. The first integrated
  subdivision-guide path was removed after it produced collapsed materials.
- **`1024_cascade` — DONE.** Full high-resolution path: LR flow (512 model) →
  `trellis2_shape_dec_upsample(×4)` → quantize+dedup to the 64³ HR scaffold →
  HR flow (1024 model, cond_1024) → 1024³ decode → dual-grid mesh. Enabled by
  flash attention (`sdpa_auto`) for the HR token counts. Validated by
  `test_cascade`: upsample coords + HR scaffold exact (to 1 voxel of ~995k),
  HR flow forward rel-L2 3.1e-4 (CPU). The 1024³ decode is the same decoder
  validated exactly at the 512 tier; its explicit parity gate is behind
  `TRELLIS2_CASCADE_DECODE` because it transiently needs ~14 GB host RAM.
- **`1536_cascade` — IMPLEMENTED, NOT YET MEASURED.** `T2_PIPE_1536` /
  `T2_CAP_1536` reuse the 1024 checkpoints on a 96³ HR scaffold, with upstream's
  token-reduction loop (`src/cascade_tokens.h`): quantize at the requested
  resolution, and while the deduplicated scaffold would reach `max_num_tokens`,
  step down by 128 with 1024 as the floor. The achieved resolution reaches the
  host as a second `T2_STAGE_UPSAMPLE` progress event and as
  `t2_mesh_grid_resolution`, so a reduced run is never silent. `T2_PIPE_AUTO`
  deliberately still stops at 1024. Unit-gated by `test_cascade_tokens` and, on
  the real reference coordinates, by `test_cascade`. **Measured on the R9700
  (2026-08-13):** runs at full 1536³ with no budget reduction on 2 of 2 objects;
  315 s total, of which the HR flow is 122.6 s and the texture stage 117.7 s —
  the 1536³ decode is only 20.2 s, ~1.6× the 1024³ one, and VRAM stays under
  16 GB of 32. So the cost is time, not memory, and `max_num_tokens` should
  *not* rise (a near-ceiling scaffold already spends ~10 min in the HR flow).
  **Quality holds in most cases, with one measured failure mode.** Across five
  runs the non-manifold edge fraction is 0.36–1.03 % regardless of tier, and the
  cleanest mesh of all is a 1536 one (0.357 %, better than every 1024 run). The
  exception is a 9.7M-voxel 1536 mesh at **7.47 %**, torn enough that the
  materials follow it down. Its signature is visible before mesh extraction: the
  finest-level expansion ratio L4/L3 hits 4.52 where every clean run sits at
  4.00–4.04, i.e. the subdivision thickens instead of following a surface —
  the RoPE-extrapolation risk landing on the subdivision head. Cost scales with
  bounding-cube occupancy, not with the tier: a slim subject at 1536 (326 s) is
  cheaper than a wide one at 1024 (345 s). `T2_PIPE_AUTO` stopping at 1024
  remains right, because the failure is not predictable from the request alone.
  Numbers in `docs/progress/1536-cascade_phase3-measure.md` and
  `docs/progress/1536-cascade_backend-limits.md`. **Still open:** where between
  3.5M and 9.7M voxels the subdivision starts to run away, and the export path's
  host-RAM high-water (D5).
- **Still to do (low VRAM):** the
  `<8 GB` case is now partly covered: the shape decoder auto-frees the flow DiTs
  around a GPU decode (see below), so the pieces of the `T2_LOAD_LOW_VRAM`
  lifecycle exist; a full per-stage load/free mode would extend that to the flow
  DiTs themselves for cards that can't hold even one 1.3B DiT + a decode.
- **Background removal** (BiRefNet/RMBG-2.0) — the demo instructs a transparent
  PNG and uses the image as-is otherwise. A separate ~1 GB seg model.
- **GPU shape decoder — DONE.** The mask-conv change plus the VRAM-aware
  placement + free/reload lifecycle (both described above) make the decoder run
  on the GPU automatically when it fits: 512³ ~2.5 s and 1024³ ~12.5 s vs ~49 s /
  minutes on the CPU. No env var required; `TRELLIS2_SHAPE_DEC_{GPU,CPU}` still
  force it, and it stays on the CPU on cards too small or in a CPU-only build.
- **CUDA CONV_3D** — the SS *occupancy* decoder (`ss_dec`, 3 s) uses a genuine
  dense `ggml_conv_3d_direct` with no CUDA kernel, so it stays on CPU regardless.
- **Quantized shipping variants / benchmarks / CI** — f16 is the demo default;
  q8/q4 conversion + a benchmark table + a two-tier CI workflow are the natural
  next hardening steps (privacy-filter.cpp has the template).
- **CPU-only SLAT sampler tightness in ctest** — the `slat` ctest is labeled
  slow and validated in-container; the sampler validates tightly on CPU
  (TRELLIS2_SLAT_STRICT=1) but that CPU run is minutes-long.
