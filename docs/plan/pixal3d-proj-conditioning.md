# Pixal3D projection conditioning (`image_attn_mode = "proj"`)

- **Status:** planned — revised 2026-08-16, see *Update 2026-08-16*
- **GitHub Issue:** none — tracked by plan key
  (`pixal3d-proj-conditioning`), so progress entries are
  `docs/progress/pixal3d-proj-conditioning_<phase>-<short-name>.md`.
- **Branch:** `pixal3d-proj-conditioning` / optional
- **Area:** trellis2 (DiT forwards, samplers) / trellis2_capi / scripts
  (converters, reference dumps) / tests / server / docs
- **Date:** 2026-08-13, revised 2026-08-16
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

## Update 2026-08-16 — what changed since this was written

Four things. The first changes the *approach*, not the detail; the
third is the reason to be careful about what this plan is expected to
deliver.

**1. The licence premise this project has worked under is wrong, and it
needs a decision before Phase 2.** `AGENTS.md` forbids adopting code
from projects "whose license is incompatible with the project" and
extends that "explicitly … to the original TRELLIS reference as well:
what is reproduced is the *behavior* …, not the source text."

The stated rationale does not apply. Both references on disk are MIT:

| Source | Licence |
|---|---|
| `docs/ref/TRELLIS-2/LICENSE` | MIT, Copyright (c) Microsoft Corporation |
| `docs/ref/TRELLIS-2_rocm/LICENSE` | MIT, Copyright (c) Microsoft Corporation |
| `docs/ideas/Pixal3D/LICENSE` | MIT, Copyright (c) 2026 Tencent |

Pixal3D's own `NOTICE` independently lists TRELLIS.2 under "Open Source
Software Licensed under the MIT License", alongside Direct3D-S2 and
MoGe; only dinov2 sits under Apache-2.0.

So the clean-room constraint is either a factual error about the licence
or an unstated policy choice (wanting the port to be independent work
rather than a derivative). **That is the reviewer's call, not an
agent's**, and it is recorded here as D0 rather than assumed either way,
because it changes the size of this plan substantially: direct
derivation with attribution is a different exercise from reproducing
behaviour from observation. Note that a repository's code licence says
nothing about its published *weights*; that is a separate check before
redistributing anything converted.

**2. The references moved and multiplied.** This plan cites
`docs/ideas/TRELLIS-2_rocm/`. There are now three copies of the TRELLIS.2
source and they are not interchangeable:

| Path | What it is |
|---|---|
| `docs/ref/TRELLIS-2/` | vanilla upstream |
| `docs/ref/TRELLIS-2_rocm/` | the ROCm-adapted fork — byte-identical to `docs/ideas/TRELLIS-2_rocm/` |
| `trellis2/` (repo root) | the working copy used by `scripts/ref_*.py`, gitignored |

Citations below still say `docs/ideas/TRELLIS-2_rocm/`; they resolve to
the same bytes as `docs/ref/TRELLIS-2_rocm/`. New work should cite
`docs/ref/`, and should say **which** of the two it means, because the
vanilla/ROCm diff is itself a finding (point 3).

**3. Pixal3D does not address what the 1024 tier is currently failing
at, and must not be sold as if it did.** The backend investigation of
2026-08-16 (`docs/plan/backend-parity.md`, Phase 0b) established that
ROCm reproduces the CPU at 512 to ten vertices and diverges at 1024,
because the HR cascade is forced onto `ggml_flash_attn_ext` — exact
attention cannot allocate its score matrix at those token counts — and
ggml's HIP flash kernels accumulate in F16 regardless of
`ggml_flash_attn_ext_set_prec()`. Measured: 89.8 % latent sign agreement
against 99.9 % exact, on a `to_subdiv` head thresholded at zero.

Projection conditioning changes *what the DiT is told about the image*.
It does not change which attention kernel the HR cascade runs on. A
Pixal3D port on this backend would inherit the holes, the tears and the
"no children at level 0" aborts unchanged. Better conditioning may well
make the model more robust to a noisy latent, but that is a hypothesis,
not a mechanism, and this plan should not be started in the expectation
that it fixes the backend.

**4. The vanilla/ROCm upstream diff is a ready-made checklist for
`backend-parity`, and it corroborates this port's findings.** Upstream
hit the same hardware and shipped four workarounds:

| Workaround | Where | What it says |
|---|---|---|
| `ATTN = 'sdpa'` forcing `SDPBackend.MATH` | `modules/sparse/attention/full_attn.py` | flash is unusable on gfx1201; upstream falls back to exact materialized attention **unconditionally**, where this port size-gates it |
| `ROCM_SAFE_CHUNK = 524_288` | `modules/sparse/linear.py` | same defect as `docs/bugs/ggml-rocm-mul-mat-column-limit.md`, same threshold as the measured f32 cap, but reported as **NaN** rather than silent zeros |
| `ROCM_SAFE_SPCONV` | `modules/sparse/config.py`, `conv/conv_flex_gemm.py` | chunked im2col + `torch.mm` instead of the Triton kernels above the same threshold |
| `conv_none.py` | `modules/sparse/conv/` | a native-PyTorch sparse conv, "MUCH SLOWER … but numerically stable" |

Two of these are new information for this project. The NaN symptom makes
`NaN > 0.0f == false` the leading explanation for the "no children at
level 0" abort. And upstream choosing exact attention *unconditionally*
is a stronger position than this port's 1 GiB gate — worth reconciling
in `backend-parity`, not here.

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
- **The 1536 cascade tier.** Pixal3D defaults to `1536_cascade`, but the
  tier is a TRELLIS.2 feature, independent of projection conditioning
  and reachable with the checkpoints already converted. It has its own
  plan, `docs/plan/1536-cascade.md`. This plan targets the 1024 tier and
  inherits 1536 once that one lands.

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
| HR scaffold quantization | `.int()`, factor `grid_res` | **`.round()`, factor `grid_res - 1`** |
| Cascade default | 1024 | 1536 (own plan, see non-goals) |
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
- [ ] `T2_CAPI_ABI_VERSION` bumped and propagated to
      `server/engine.go:20`; verified against the built DLL. The plan was
      written against 15 → 16; the header is at **17** as of 2026-08-16
      (`t2_run_config`), so this lands on 17 → 18 unless something else
      bumps it first. Read the header, do not trust this line.
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
| `docs/ideas/Pixal3D/` | the Pixal3D reference implementation (MIT, Tencent) |
| `docs/ref/TRELLIS-2/` | vanilla TRELLIS.2 upstream (MIT, Microsoft) — the baseline Pixal3D was built from |
| `docs/ref/TRELLIS-2_rocm/` | the ROCm-adapted fork; its diff against vanilla is the gfx1201 workaround checklist (see Update, point 4) |
| `THIRD_PARTY.md`, `docs/VERIFICATION.md`, `server/engine.go` | obligations from the criteria above; `THIRD_PARTY.md` also carries D0's attribution if derivation is chosen |

## Architecture decisions

**D0 — Derivation or clean room: undecided, blocks Phase 2.** Added
2026-08-16 with the licence finding above. The two readings lead to
different work:

- *Clean room (status quo).* Behaviour is reproduced from observation
  and reference dumps; no source text is carried over. This is what
  every existing stage did, it is unambiguously safe, and it is the
  reason Phase 0 exists at all — the dumps are the only channel through
  which the reference speaks.
- *Direct derivation.* MIT permits translating the reference's Python
  into this repository's C++ provided the copyright notice travels with
  it: an entry in `THIRD_PARTY.md`, attribution in the touched files,
  and Microsoft's and Tencent's notices retained. Phases 1, 3 and 5
  shrink considerably — `ProjGrid`, `ProjectAttention` and the NAF
  neighbourhood attention are then transcription against a known-good
  original rather than reconstruction.

The recommendation is to settle this **before** Phase 2, not during: a
converter written under one assumption and reviewed under the other is
wasted work. Whichever is chosen, the parity gates in this plan do not
change — a derived implementation still has to prove it reproduces the
reference numerically, because a translation can be wrong in ways
reading cannot catch.

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
      (LR‖HR concat), on the existing 1024 tier. Whether the HR scaffold
      quantization has to follow Pixal3D's rounding rather than
      TRELLIS.2's truncation is decided here (see the open question
      below), because it changes which voxels the HR flow ever sees.
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

- **This plan cannot deliver the quality the 1024 tier is currently
  losing.** See Update point 3. The failures being attributed to the
  model on this hardware — holes, tears, `no children at level 0` — are
  the HR cascade's flash attention, and `proj` conditioning runs through
  the same kernel. If Pixal3D is started as a response to those
  symptoms, it will not resolve them and the effort will read as a
  failure of this plan rather than of the expectation. `backend-parity`
  Phase 0b is the plan that owns that problem.
- **D0 is unresolved and gates Phase 2.** Clean-room reproduction and
  direct MIT derivation are both defensible; starting the converters
  before the answer risks rewriting them.
- **Phase 0 is the real risk, not NAF.** The whole plan is
  parity-gated, and the gate needs a reference container in which
  `natten` builds. If that fails, Phase 1 and 3 can still proceed
  against hand-checked geometry, but Phase 4 cannot be validated.
- **fp32 storage is inferred** from the published blob sizes, not read
  from the safetensors header. The first converter run settles it.
- **NAF's RoPE may not match the DINOv3 axial formulation** despite the
  shared base of 100.0 (D5).
- **Pixal3D quantizes the HR scaffold differently.** TRELLIS.2 uses
  `((c + 0.5) / lr_res * grid_res).int()`
  (`docs/ideas/TRELLIS-2_rocm/.../trellis2_image_to_3d.py:415`), which
  `src/trellis2_capi.cpp:1003` reproduces exactly. Pixal3D uses
  `((c + 0.5) / lr_res * (grid_res - 1)).round()`
  (`docs/ideas/Pixal3D/pixal3d/pipelines/pixal3d_image_to_3d.py:717`) —
  both the rounding mode and the scale factor differ, so the two produce
  different HR coordinate sets from the same LR SLAT. Whether this is a
  deliberate change the Pixal3D weights depend on, or incidental drift,
  is unresolved. It has to be settled before Phase 5, and it means the
  quantization may need to branch on the conditioning mode rather than
  being shared (cf. D2 in `docs/plan/1536-cascade.md`).
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
