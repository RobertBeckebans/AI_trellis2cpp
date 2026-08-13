# Pixal3D projection conditioning (`image_attn_mode = "proj"`)

- **Status:** planned
- **GitHub Issue:** none — tracked by plan key
  (`pixal3d-proj-conditioning`), so progress entries are
  `docs/progress/pixal3d-proj-conditioning_<phase>-<short-name>.md`.
- **Branch:** `pixal3d-proj-conditioning` / optional
- **Area:** trellis2 (DiT forwards, samplers) / trellis2_capi / scripts
  (converters, reference dumps) / tests / server / docs
- **Date:** 2026-08-13
- **Author:** AI (prepared for review)

## Goal

TRELLIS.2 injects the image only through cross-attention: the DiT sees
1029 DINOv3 tokens as a permutation-invariant K/V set with no positional
relation to the voxel it is denoising. Pixal3D
([TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D), SIGGRAPH
2026, MIT) replaces that with an explicit pixel-to-voxel correspondence:
every voxel of the DiT's own token grid is projected into the image
through a camera, its DINOv3 feature is sampled there, and the result is
added to each block's cross-attention output. The reported gain is
fidelity — the geometry stops drifting away from what the image actually
shows.

For this port the decisive property is that **the denoiser architecture
does not change at all**. The released Pixal3D checkpoints are the same
1.3B DiTs this repository already implements, finetuned in the same
latent space, plus one linear layer per block. The reference
implementation is unpacked under `docs/ideas/Pixal3D/`, the TRELLIS.2
baseline it was built from under `docs/ideas/TRELLIS-2_rocm/`.

**Goal:** a second, selectable conditioning path — the existing
TRELLIS.2 GGUFs keep working unchanged; loading the Pixal3D GGUFs
instead switches the pipeline into `proj` mode.

## Non-goals

- **Multi-image conditioning.** This was the question that started the
  investigation, and the answer is no. Pixal3D's latent space is
  *view-aligned*: training objects are pre-rotated into the conditioning
  camera's frame, so `ProjGrid.forward()` carries a hard
  `assert transform_matrix is None`
  (`docs/ideas/Pixal3D/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:211`)
  and the dataset explicitly documents why
  (`.../datasets/components.py:274`). Two views define two mutually
  inconsistent target frames. The `"num_views": 2` in the configs is
  `SparseStructureLatentView` picking *one* of two view-aligned latent
  encodings at random (`.../datasets/sparse_structure_latent.py:380`) —
  augmentation, not multi-image. `MultiImageConditionedMixin` exists
  (`.../datasets/components.py:292`) but no `proj` config references it.
  Supporting real multi-view would mean retraining, which is out of
  scope for an inference port.
- **Any training or finetuning.** Everything here is downloads plus
  conversion.
- **Replacing the TRELLIS.2 path.** Both coexist; the loaded GGUF set
  decides.
- **The `gated_proj` mode** (DINOv3 + Flux VAE, `GatedProjectAttention`).
  No released checkpoint uses it.
- **MoGe-2 for FOV estimation.** See D6 — a manual FOV is the first
  iteration.

## What actually differs from the current port

Verified against the reference configs and the published checkpoint
sizes:

| | TRELLIS.2 (current port) | Pixal3D |
|---|---|---|
| DiT hparams (res/in/out/ch/blocks/heads/mlp/rope/share_mod/qk_rms) | `trellis2_ss_flow_hparams`, `src/trellis2.h:99` | **identical** (`configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json`) |
| `image_attn_mode` | `cross` | **`proj`** |
| Cross-attention context | 1029 DINOv3 tokens | **5 tokens** (CLS + 4 register) |
| Per-block extra | — | **`proj_linear`**: `[proj_in_channels → 1536]` + bias |
| `proj_in_channels` | — | 1024 (SS), 2048 (shape 512/1024, tex 1024) |
| SLAT normalization (`pipeline.json`) | mean/std, 32 ch | **byte-identical** |
| Sampler params (steps/guidance/interval/rescale) | 12 / 7.5 / … | **identical** |
| `shape_dec_next_dc_f16c32_fp16.safetensors` | 948,490,494 B | **948,490,494 B** |
| `tex_dec_next_dc_f16c32_fp16.safetensors` | 948,458,812 B | **948,458,812 B** |
| `ss_flow_img_dit_1_3B_64_bf16.safetensors` | 2,584,426,920 B | 5,359,822,584 B |
| `slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` | 2,584,574,424 B | 5,546,764,048 B |
| Image preprocess bbox padding | `size * 1` | **`size * 1.1`** |
| Cascade | 512 / 1024 | 1024 / **1536**, adaptive token budget |
| Extra models | — | **NAF** feature upsampler (Apache-2.0, 2.7 MB) |

Two numbers carry most of the argument:

- The decoders are **byte-identical in size** and the normalization
  arrays match exactly, so Pixal3D finetuned only the flow DiTs, inside
  the unchanged SLAT latent space. `ggufs/ss_dec_*`, `ggufs/shape_dec_*`
  and `ggufs/tex_dec_*` are reusable **as-is**.
- Doubling the TRELLIS.2 flow size (Pixal3D stores fp32 despite the
  `_bf16` filename — inferred from size, to be confirmed by the first
  converter run) leaves a remainder of 377,615,200 B on both the shape
  and the texture 1024 checkpoint. Expected for
  30 × (2048 × 1536 + 1536) fp32 parameters: 377,671,680 B. That is the
  `proj_linear` stack and nothing else.

`ProjectAttention` in full
(`docs/ideas/Pixal3D/pixal3d/modules/attention/proj_attention.py:45`):

```python
global_out = self.cross_attn_block(x, global_context)   # 5 tokens
proj_out   = self.proj_linear(proj_context)             # one vector per DiT token
return proj_out + global_out
```

## Acceptance criteria

- [ ] `trellis2_ss_flow_forward` and `trellis2_slat_flow_forward` accept
      a `proj` context and reproduce the reference forward within the
      gate used for the existing flow taps (rel-L2 ≈ 3e-4 on CPU).
- [ ] The back-projection (`ProjGrid` equivalent) reproduces the
      reference `image_points` / `valid_mask` for the SS, shape and
      texture grid resolutions — this is pure geometry and should be
      near-exact, not merely within tolerance.
- [ ] The NAF upsampler reproduces the reference `hr_features` at the
      sampled positions within the DINOv3 gate (rel-L2 ≤ 1e-5), and the
      sparse evaluation of D4 is shown to be *exact* against a full
      dense NAF pass, not an approximation.
- [ ] A Pixal3D generation runs end to end through `t2_generate` and
      produces a mesh visibly closer to the input image than the
      TRELLIS.2 path on the same seed. Recorded as a side-by-side in
      `docs/progress/`.
- [ ] Loading the existing TRELLIS.2 GGUFs still selects the `cross`
      path and every existing tap in `docs/VERIFICATION.md` is
      unchanged. This plan must not move a single existing number.
- [ ] `docs/VERIFICATION.md` gains rows for: back-projection, NAF,
      `proj` SS-flow forward, `proj` SLAT-flow forward.
- [ ] `T2_CAPI_ABI_VERSION` bumped 15 → 16 and propagated to
      `server/engine.go:20`; verified against the built DLL.
- [ ] NAF listed in `THIRD_PARTY.md` with its Apache-2.0 NOTICE
      obligation honoured.
- [ ] `ctest --test-dir build -C Release -LE model` green.
- [ ] No `SKIP` (return code 77) counted as a pass anywhere in this
      plan's verification.

## Context / affected files

| Path | Why relevant |
|---|---|
| `src/trellis2.h` | new hparams (`proj_in_channels`), new cond struct, NAF model handle, camera params |
| `src/trellis2.cpp:522` | `trellis2_ss_flow_forward` — cross-attn context length, `proj_linear` add |
| `src/trellis2.cpp:2176` | `trellis2_slat_flow_forward` — same, sparse |
| `src/trellis2.cpp:777`, `:2349`, `:2465` | the three samplers — carry `proj` through CFG (neg branch is zeros, as today) |
| `src/trellis2_capi.h:96` | `t2_pipeline_load` — NAF GGUF path, proj-mode capability bit |
| `src/trellis2_capi.h:130` | `t2_generate` — FOV parameter |
| `src/trellis2_capi.cpp:784`, `:1029` | where the DINO cond is built today |
| `src/trellis2_capi.cpp:~886` | the 64³ → 32³ scaffold; the proj grid attaches to these coords |
| `scripts/convert_ss_flow_to_gguf.py`, `convert_slat_flow_to_gguf.py`, `convert_tex_flow_to_gguf.py` | `proj_linear.*`, fp32 input |
| `scripts/download_models.sh` | Pixal3D repo + `naf_release.pth` |
| `scripts/convert_all.sh` | proj GGUF variants |
| `docs/ideas/Pixal3D/` | the reference implementation |
| `THIRD_PARTY.md`, `docs/VERIFICATION.md`, `server/engine.go` | obligations from the criteria above |

## Architecture decisions

**D1 — A parallel conditioning path, not a replacement.** The `proj`
weights and the `cross` weights are different checkpoints; the GGUF KV
store carries `proj_in_channels` (absent or 0 ⇒ `cross`). The forward
branches on it. This keeps every existing tap in `docs/VERIFICATION.md`
untouched, which the acceptance criteria require.

**D2 — `proj` context is a per-token buffer, not an attention input.**
`proj_linear(proj_ctx)` is a single `ggml_mul_mat` over the DiT's own
token axis plus an add onto the cross-attention result. It is *not*
another K/V set. Cost per block: one `[proj_in × 1536]` matmul over N
tokens. Simultaneously the cross-attention K/V drops from 1029 to 5
tokens, so the `proj` forward should be measurably **cheaper** than the
current one — worth measuring rather than assuming.

**D3 — Project only the active voxels.** The reference builds the dense
R³ proj grid and then indexes it at `coords`
(`docs/ideas/Pixal3D/pixal3d/pipelines/pixal3d_image_to_3d.py:275-281`).
At the HR stage that is 64³ or 96³ entries for a few tens of thousands
of active voxels. The port projects the active coordinates directly.
Same values, no dense intermediate.

**D4 — Evaluate NAF sparsely.** NAF's three parts — a 2-layer conv image
encoder, an adaptive average pool of the LR features, and a 9×9
neighborhood attention — are all local. Evaluating it only at the
projected sample positions (plus the four bilinear neighbours and a
convolution halo) is therefore **exact**, not an approximation. This
turns a 1024×1024 × 81 × 1024-channel upsample into a per-voxel gather.
The acceptance criteria require this exactness to be *demonstrated*
against a dense pass, because the argument rests on the halo being
handled correctly.

**D5 — Neighborhood attention as an own op.** `natten` is the
reference's convenience, not an architectural requirement: 9×9 means 81
keys per query, a gather plus softmax. NAF's RoPE uses `rope_base=100.0`,
the same base as the DINOv3 encoder already ported
(`src/trellis2.h:562`); whether the axial formulation is identical must
be checked before reusing that code, not assumed.

**D6 — Manual FOV first.** The camera is fully described by
`camera_angle_x` and `distance`, with `distance` derived from the FOV so
the image corners meet the unit cube's back face
(`docs/ideas/Pixal3D/inference.py:124`). `ProjGrid` only ever uses the
canonical front view. MoGe-2 estimates the FOV in the reference, but
`--fov 0.2` is the documented fallback and is one float in the C API.
Porting MoGe-2 is a separate project; if the fixed FOV proves too
lossy, that becomes a follow-up plan.

**D7 — The 1.1× crop is part of the model.** Pixal3D pads the alpha
bounding box by 10 % (`size = int(size * 1.1)`) where TRELLIS.2 uses
`size * 1`. Under `cross` conditioning a different crop merely shifts
features; under `proj` it shifts **the projection geometry**, so a
1:1 reproduction of the crop is now correctness-critical, not cosmetic.
`trellis2_preprocess_rgba` (`src/trellis2.h:635`) needs a padding
parameter. Pixal3D's composite onto `bg_color` defaults to black and is
then arithmetically identical to the existing premultiply.

**D8 — Reuse the decoders.** Byte-identical checkpoint sizes and
identical normalization arrays; `ggufs/ss_dec_*`, `shape_dec_*`,
`tex_dec_*` stay as they are. Note that Pixal3D ships **no** `shape_enc`
/ `tex_enc` and no 512 texture flow, so the standalone
arbitrary-mesh texturing path (`trellis2_shape_enc_*`) keeps using the
TRELLIS.2 weights.

## Plan (phases)

- [ ] **Phase 0 — Reference environment and fixtures.** Extend
      `docker/Dockerfile.ref` for the Pixal3D reference (`natten==0.21.0`
      needs `NATTEN_CUDA_ARCH` and builds from source — this is the
      phase most likely to hurt). Write `scripts/dump_pixal3d_reference.py`
      producing taps for: `ProjGrid` output (`image_points`, `depth`,
      `valid_mask`, sampled `z_proj_lr`), NAF `hr_features`, `z_global`,
      and one `proj` forward per DiT. Nothing downstream can be
      parity-gated before this exists.
- [ ] **Phase 1 — Preprocessing, camera, back-projection.** The 1.1×
      crop variant (D7), the FOV → distance derivation, the pinhole
      projection of voxel centres and the bilinear sample from the
      DINOv3 patch map. Pure C++, no ggml. Gated against the Phase 0
      geometry taps.
- [ ] **Phase 2 — Converters.** `proj_linear.{weight,bias}` per block,
      fp32 source handling, `proj_in_channels` into the GGUF KV store.
      Extend `scripts/download_models.sh` (Pixal3D repo, ~24 GB) and
      `scripts/convert_all.sh`.
- [ ] **Phase 3 — `ProjectAttention` and the SS stage.** Branch both
      flow forwards on `proj_in_channels`; 5-token cross-attention plus
      the per-block add. The SS stage uses **no NAF**
      (`docs/ideas/Pixal3D/inference.py:27-31`), so this phase already
      yields a complete, parity-gated `proj` sparse structure — the
      first real end-to-end signal, before NAF exists.
- [ ] **Phase 4 — NAF.** GGUF conversion of `naf_release.pth`
      (2,664,431 B), the neighborhood-attention op (D5), sparse
      evaluation and its exactness proof (D4). `THIRD_PARTY.md` entry.
- [ ] **Phase 5 — Shape and texture cascade.** Shape 512 → upsample ×4 →
      quantized HR scaffold → shape HR → texture, with `proj_in = 2048`
      (LR‖HR concat). The adaptive token budget (`max_num_tokens =
      49152`, shrinking the HR resolution in steps of 128) and the 1536
      cascade come last, since the fork is currently fixed at 1024.
- [ ] **Phase 6 — C API, server, docs.** NAF GGUF path and proj
      capability bit in `t2_pipeline_load`, FOV in `t2_generate`, ABI
      15 → 16, `server/engine.go:20`, `README.md`,
      `docs/VERIFICATION.md` rows.

## Tests / verification

- `tests/test_proj_grid.cpp` — back-projection and crop against the
  Phase 0 geometry taps. No GPU, no large fixtures; belongs in the
  `-LE model` set.
- `tests/test_naf.cpp` — dense NAF against the reference tap, **and**
  sparse-vs-dense equality (D4).
- `tests/test_proj_flow.cpp` — `proj` forward for both DiTs against the
  reference, mirroring `test_ss_flow_forward` / `test_slat`.
- `ctest --test-dir build -C Release -LE model` — must stay green
  throughout; a `SKIP` is not a pass.
- `ctest --test-dir build -C Release` — full parity once the Phase 0
  fixtures exist.
- Manual smoke test: same image, same seed, TRELLIS.2 GGUFs vs Pixal3D
  GGUFs, side by side in `docs/progress/`.

## Risks / open questions

- **Phase 0 is the real risk, not NAF.** The whole plan is
  parity-gated, and the gate needs a reference container in which
  `natten` builds. If that fails, Phase 1 and 3 can still proceed
  against hand-checked geometry, but Phase 4 cannot be validated.
- **fp32 storage is inferred** from the published blob sizes, not read
  from the safetensors header. The first converter run settles it.
- **NAF's RoPE may not match the DINOv3 axial formulation** despite the
  shared base of 100.0 (D5).
- **Halo correctness in the sparse NAF evaluation** (D4) — the internal
  bilinear downsample when the guide exceeds 4× the target size is the
  part most likely to be got wrong.
- **The fixed FOV** (D6) may visibly distort on wide-angle inputs; the
  reference ships MoGe-2 for exactly that reason.
- **VRAM.** The reference reports ~18 GB standard / ~10–12 GB low-VRAM
  for a 4B-scale cascade, and the fp32 checkpoints are twice the size on
  disk. The existing lazy per-stage loading should carry over, but the
  1536 cascade at 49,152 tokens is untested here.
- **No 512 texture flow and no encoders** in the Pixal3D release (D8) —
  a Pixal3D pipeline cannot currently serve the standalone texturing
  API.

## Release note

Optional Pixal3D projection conditioning: an alternative image-cond path
that back-projects DINOv3 features onto the voxel grid instead of
relying on cross-attention alone, for markedly higher fidelity to the
input image. Selected by loading the Pixal3D GGUFs; the TRELLIS.2 path
is unchanged.

## Proposed commit message

```text
Condition the DiTs on back-projected image features, not just attention
```
