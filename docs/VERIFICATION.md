# Verification

Every ported stage is validated numerically against the PyTorch reference, per
component, not just end-to-end — the depth-anything.cpp approach. Reference
activations are dumped from `microsoft/TRELLIS.2` (via `scripts/refgen.sh`
inside the CUDA container) into GGUF files under `dumps/`, and the C++ side is
compared tap-by-tap with `tests/parity.hpp` (gate: `|got-ref| <= atol +
rtol*|ref|`, reported as max-abs / rel-L2 / per-row).

## One-time setup

```sh
docker build -f docker/Dockerfile.ref -t trellis2-ref docker   # PyTorch reference env
scripts/download_models.sh                                     # HF checkpoints -> models/
docker run --rm -v "$PWD":/work -w /work trellis2-ref bash -c '
  python convert_dino_to_gguf.py        --output ggufs/dino_f32.gguf       --ftype 0
  python convert_ss_flow_to_gguf.py     --model models/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors --output ggufs/ss_flow_f32.gguf --ftype 0
  python convert_ss_dec_to_gguf.py      --model models/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors --output ggufs/ss_dec_f32.gguf --ftype 0
  python convert_slat_flow_to_gguf.py   --model models/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors --pipeline-json models/TRELLIS.2-4B/pipeline.json --output ggufs/slat_flow_f32.gguf --ftype 0
  python convert_shape_dec_to_gguf.py   --output ggufs/shape_dec_f32.gguf  --ftype 0'
scripts/refgen.sh                                              # dump reference activations
```

The container is CUDA-only. On a host whose GPU it cannot use — the AMD
development machine — the same scripts run natively against the host's torch
build instead; see [`reference-environment.md`](reference-environment.md) for
the `uv` setup, and read its D1 warning before trusting a numeric tap produced
that way.

## Run

```sh
cmake -B build -DTRELLIS2_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build -LE model        # fast, no assets (marching cubes, preprocess)
ctest --test-dir build                  # full parity (needs ggufs/ + dumps/)
```

## Parity table (f32 GGUF vs true-fp32 reference)

| stage | test | tap coverage | result |
|---|---|---|---|
| image preprocess (alpha crop, premultiply, Lanczos-512) | `test_preprocess` | full 512×512 RGB | **byte-exact** (0/786432 differ) |
| DINOv3 ViT-L/16 encoder | `test_dino` | 40 taps: embeddings, RoPE, per-layer output + first/last-layer detail (norm/attn/layerscale/mlp), final affine-free LN | **PASS**, rel-L2 ≤ 7e-7 all taps |
| SS-flow DiT forward | `test_ss_flow_forward` | full output | **PASS**, rel-L2 2.4e-4 |
| SS-flow Euler sampler (12-step CFG) | `test_ss_sample` | z_s latent | **PASS**, rel-L2 5.7e-3, sign 99.85% (CPU) |
| SS decoder (dense 3D-conv → 64³ occupancy) | `test_ss_dec` | occupancy logits | **PASS**, rel-L2 2e-5 |
| shape-SLAT flow forward | `test_slat` | full output | **PASS**, rel-L2 2.9e-4 (CPU) / 8e-4 (GPU) |
| shape-SLAT VAE decoder (sparse ConvNeXt U-Net, 4 levels, 16× up) | `test_slat` | 18 taps: per-level features + subdivision logits + active sets, all 5 levels, plus the finest level's `pre_up` and the final 7-ch output | **PASS**, 18/18, rel-L2 ≤ 1.1e-06 (`lvl4.pre_up` 3.962e-07, `out7` 4.318e-07; all coord sets exact). Reference produced on the **CPU** — the same script on ROCm fails those two taps at 0.979 / 1.103, and that failure is in the reference's own convolution, not the port. See `docs/progress/rocm-native-reference_3-slat-dump-and-out7.md` |
| integrated subdivision guide | `test_slat` | all decoder levels; final guide coordinates equal decoded shape coordinates | **PASS** |
| standalone shape encoder → texture flow → texture decoder | `test_texture` | shape latent, flow forward/sampler, guided 6-channel PBR decode | parity-gated; sampler backend drift uses the documented loose gate |
| sparse PBR surface sampling | `test_pbr_sampling` | dense trilinear interpolation + sparse-boundary normalization | **PASS** |
| GLB PBR/alpha export | `test_mesh_export` | direct vertex RGBA, retained metallic/roughness, glTF alpha mode | **PASS** |
| marching-cubes preview extraction (`examples/marching_cubes.h`) | `test_marching_cubes` (invariants) | watertight-manifold, Euler characteristic, winding | **PASS** |
| dual-grid mesh extraction (`examples/flexible_dual_grid.h`) | `test_dual_grid` (invariants) | closed, 2-manifold, Euler characteristic on analytic sphere/torus at two grids; `fill_holes` limit contract | **PASS**, χ = 2 and 0 as expected, 0 boundary and 0 non-manifold edges. Previously credited to `test_marching_cubes`, which exercises a different extractor — this path had no test |
| **1024 cascade** — decoder upsample(×4) → 512³ coords | `test_cascade` | full coord set + quantized 64³ HR scaffold | **PASS**, set match to 0.0001% (1 voxel of 995k) |
| **1024 cascade** — HR (1024-model) flow forward | `test_cascade` | full output | **PASS**, rel-L2 ~3e-4 (CPU); ~1e-2 on GPU flash |
| **1024 cascade** — final 1024³ decode (3.97M voxels) | `test_cascade` | per-level features + subdivision + 7-ch output | **PASS**, rel-L2 ≤ 2e-2, set within 0.0001% |
| **cascade token budget** — quantize (truncating) + step-down loop + 1024 floor | `test_cascade_tokens` | formula on hand-checked cells; chosen resolution vs. independently derived per-grid counts for every budget; 1024 scaffold vs. an independent re-implementation of the pre-1536 inline code | **PASS**, no assets needed (runs in `-LE model`) |
| **1536 cascade** — 96³ HR scaffold + selected resolution | `test_cascade` | full coord set + `hr_resolution` | **PASS**, exact set match: 27,540 of 27,540 voxels at 96³, resolution 1536 = reference, sym-diff 0.0000%. Reference produced on ROCm (R9700) with `dump_cascade_reference.py --skip-hr-sampler`; per plan D1 that is authoritative here because both sides are the integer quantization `((c+0.5)/lr_res*grid).int()`, which has no precision question. The budget did **not** reduce on this fixture (27,540 < 49,152), so the *reduction* branch is still unreferenced |
| **1536 cascade** — end-to-end run on the R9700 | manual (`server/`, 2026-08-13) | achieved resolution, stage wall clock, mesh size, VRAM ceiling | **RUNS at full 1536³**, 2 of 2 objects, no budget reduction. 315 s total, 1536³ decode 20.2 s, 6.66M-tri mesh, VRAM ceiling < 16 GB of 32 GB. See `docs/progress/1536-cascade_phase3-measure.md` |
| **1536 cascade** — mesh topology across tiers and seeds | manual, `[mesh]` line under `TRELLIS2_TIMING` | non-manifold and boundary edge fraction of the extracted mesh | **Usually fine, one measured failure.** 4 of 5 runs land at 0.36–1.03 % non-manifold regardless of tier; the best result of all is a 1536 one (0.357 %, cleaner than every 1024 run). The outlier is a 9.7M-voxel 1536 mesh at **7.47 %**, where the finest subdivision ran away (L4/L3 = 4.52 against ~4.0 everywhere else). See `docs/progress/1536-cascade_backend-limits.md` |
| **1536 cascade** — early warning for that failure | same | finest-level expansion ratio L4/L3 in the `[shape_dec]` lines | 4.00–4.04 on every clean run, **4.52** on the broken one. Above 4 means the subdivision is thickening rather than following a surface. Flags the catastrophic case before mesh extraction; it is **not** a fine-grained quality score |
| **1536 cascade** — decode VRAM peak, host RAM high-water | — | — | **NOT INSTRUMENTED.** The < 16 GB ceiling is an external reading, so it bounds but does not pin the decode transient; `decode_vram_peak`'s 1536 entry stays the conservative extrapolation. Host RAM: generation side ~0.5 GB (derived from the mesh size), export side (plan D5) still unmeasured |

Notes:
- **Platform caveat — the table above is a CUDA/Linux measurement.** The first
  full run on Windows/HIP (Radeon AI PRO R9700, 2026-08-14) did not reproduce
  all of it; three of the five failures have since been addressed. Still open
  there: `ss_dec` cannot run on the HIP backend at all (`CONV_3D` unsupported →
  `GGML_ASSERT`; the pipeline already pins that stage to the CPU, only the test
  does not), `dino` misses its ≤ 7e-07 row on both backends (`embd` 4.6e-04,
  `cond` 1.6e-04), and `chunked_decode` fails for every non-default block size.
  Fixed: the HIP graph crash (`ss_sample`) and the flash-attention divergence
  (`ss_flow_forward`, `ss_sample`) — see the notes below. Details and the
  CPU/HIP split per test in
  `docs/progress/rocm-native-reference_1-2-cheap-chain-and-1536.md`. Treat a row
  here as validated on the backend it was taken on, not as a platform guarantee.
- **Flash attention (size-gated, not unconditional).** `sdpa_auto()` uses the
  exact materialized `[L_k, L_q, heads]` softmax while that matrix stays under
  1 GiB, and `ggml_flash_attn_ext` above it. In practice: SS-flow (~0.8 GiB) and
  512-SLAT (~0.4 GiB) are exact; both HR cascade tiers (~7.7 GB at 1024,
  >100 GB at 1536) are flash, where exact is not an option at all.
  `TRELLIS2_SDPA_EXACT` / `TRELLIS2_SDPA_FLASH` pin one path,
  `TRELLIS2_SDPA_EXACT_MAX_MB` moves the gate (`0` = always flash). A free-VRAM
  check can only *lower* exact usage, so a card that cannot afford the matrix
  degrades to flash rather than failing to allocate; it prints a note when it
  does, because it makes the numerics depend on machine state.

  Flash is not free on GPU: ggml's CUDA/HIP kernels accumulate in F16 and ignore
  `ggml_flash_attn_ext_set_prec()` outright — nothing under `ggml-cuda` reads
  `GGML_PREC_F32` for this op, so the port's request goes nowhere. On CPU it is
  bit-faithful, so the choice only matters on GPU. Measured on ROCm/gfx1201,
  identical for f16 and f32 weights (i.e. the weight format does not mask it):

  | | flash | exact |
  |---|---|---|
  | SS-flow forward | 5.198e-02 | 3.050e-06 |
  | 12-step sampler | 2.847e-01 | 2.135e-03 |
  | latent sign agreement | 89.8 % | 99.9 % |

  About one latent voxel in ten changes sign under flash, which is the coarse
  occupancy structure rather than a rounding digit — hence the gate. Note this
  also revises the "sampler drift" note below: with exact attention the GPU
  reproduces the CPU sampler (99.9 % against the documented 99.85 %), so the
  ~0.1 rel-L2 previously attributed to chaotic amplification was the flash
  kernel. It was flash-unconditional for one release, on the grounds of ~30 %
  speed and the exact path's `alloc_graph` failures under VRAM pressure; the
  VRAM check now covers that case without costing accuracy elsewhere. See
  docs/PLAN.md for the per-stage runtime profile.
- **TF32 matters.** PyTorch's default CUDA matmul/attention uses TF32 (≈10-bit
  mantissa) and reduced-precision flash SDPA, which shows up as ~1e-3 relative
  error versus true fp32. `scripts/ref_common.py` disables it so the golden
  dumps are real fp32; otherwise a correct port looks like it has a 0.08-rel-L2
  bug (this exact trap cost a debugging session — see the flow-forward gate).
- **Reference device matters.** `scripts/ref_common.py` does not use an upstream
  conv backend — it registers its own sparse convolution and hard-assigns
  `config.CONV`, so the `[SPARSE]` banner naming a backend says nothing about
  what ran. That convolution is wrong on ROCm: 4.9x too large, uncorrelated with
  the same arithmetic on the CPU, and not reproducible run to run. A reference
  generated there fails the decoder's finest taps and reads as a *port* defect,
  which sends the search to the wrong side. Numeric reference dumps go on the
  CPU. Leading suspect is that its per-offset GEMM is unchunked and exceeds the
  524,288-row limit the ROCm fork observes everywhere else
  (`trellis2/modules/sparse/linear.py`); unconfirmed. This does not apply to the
  1536 row above, where both sides compute the same integer quantization.
- **Sampler drift.** The 12-step Euler + CFG-rescale loop chaotically amplifies
  per-step fp differences between backends; it validates tightly on CPU and
  drifts to ~0.1 rel-L2 on GPU. The decoder gate therefore decodes the
  *reference* SLAT so decoder parity is independent of sampler trajectory.
- **Subdivision boundary.** A handful of level-3 subdivision logits sit within
  fp-noise of zero; the `>0` threshold can flip them, so the final active-voxel
  set differs by ~4 voxels out of ~4 million (0.0001%) run-to-run and
  hardware-to-hardware. This is inherent to a hard threshold, not a port bug.

## GPU

`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` (Blackwell / RTX 50-series). The
flow DiTs and DINOv3 run on CUDA; the 3D-conv decoders (SS decoder CONV_3D and
the sparse-conv gather-GEMM) run on CPU because the bundled ggml has no CUDA
CONV_3D kernel — they are a small fraction of total inference time. GPU f32
matmul is fp16-class, so tap parity on CUDA is ~1e-3 (deterministic, not device
noise); CPU is the tight-tolerance reference backend.
