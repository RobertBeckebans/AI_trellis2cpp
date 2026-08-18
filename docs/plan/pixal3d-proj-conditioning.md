# Pixal3D projection conditioning (`image_attn_mode = "proj"`)

- **Status:** planned — revised 2026-08-16 (see *Update 2026-08-16*); D0
  resolved 2026-08-18, derivation permitted
- **GitHub Issue:** none — tracked by plan key
  (`pixal3d-proj-conditioning`), so progress entries are
  `docs/progress/pixal3d-proj-conditioning_<phase>-<short-name>.md`.
- **Branch:** `pixal3d-proj-conditioning` / optional
- **Area:** trellis2 (DiT forwards, samplers) / trellis2_capi / scripts
  (converters, reference dumps) / tests / server / docs
- **Date:** 2026-08-13, revised 2026-08-16 and 2026-08-18
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
needs a decision before Phase 2.** — **Decided 2026-08-18: derivation is
permitted, see D0.** The paragraphs below are the finding that led there
and are kept for the reasoning; the rule text they quote has since been
corrected. `AGENTS.md` forbids adopting code
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

## Update 2026-08-18 — the reference environment, and what the backend work settled

Eight things. The first five are the reference environment, which is now settled
and which deletes Phase 0 as written — one of its stated risks disappears and a
harder one takes its place. Point 6 makes this plan's central comparison
measurable for the first time. Points 7 and 8 are a criterion that cannot be met
as written, and three numbers that are simply out of date.

**1. No Docker, no NVIDIA, PyTorch through uv. Reviewer's working rule, and it
deletes Phase 0 as written.** This machine has CPU, ROCm and Vulkan and no CUDA
device anywhere in reach, so `docker/Dockerfile.ref`
(`pytorch/pytorch:2.7.1-cuda12.8`) is not an environment this project can run —
`rocm-native-reference` established that for its own phase 0 and built the
replacement: uv extras (`rocm`/`cuda`/`cpu`) in `pyproject.toml`,
`.python-version` at 3.12, `docs/reference-environment.md`, and
`scripts/ref_env_check.py` as the host gate. Phase 0 targets that path. Nothing
in this plan extends the container.

**2. The `natten` risk is worse than stated, and it is not a build problem.**
Phase 0 says `natten==0.21.0` "needs `NATTEN_CUDA_ARCH` and builds from source".
Upstream does not build it from source at all —
`docs/ideas/Pixal3D/requirements-hfdemo.txt` pins a **prebuilt wheel**,
`natten-0.21.0+torch2.6cu124-cp310-cp310-linux_x86_64.whl`, alongside the same
treatment for `flash_attn_3`, `flex_gemm`, `cumesh`, `o_voxel`, `nvdiffrast` and
`nvdiffrec_render`. Linux, x86_64, CUDA 12.4, CPython 3.10, torch 2.6. Against
Windows / 3.12 / torch 2.9.1+rocmsdk, not one of them installs, and with no
NVIDIA card none of them would run if it did.

Most of that list is irrelevant here — `cumesh`, `nvdiffrast` and
`nvdiffrec_render` are meshing and rendering, which this project does itself, and
`flash_attn`/`flex_gemm` are what the ROCm fork already replaces. `natten` is the
one that is genuinely needed, and it is needed only for NAF.

**3. NAF is not in this tree, and D4/D5 rest on a repository that is not on
disk.** `image_conditioned_proj.py:415-424` loads it with
`torch.hub.load("valeoai/NAF", "naf", pretrained=True, trust_repo=True)` — a
runtime fetch of a third party's code, not a vendored module. `docs/ideas/Pixal3D/`
contains no NAF source and no `natten` import anywhere in its Python; Pixal3D's
`NOTICE` does not list NAF among its third-party components either. So the
"Apache-2.0, 2.7 MB" in the table below is unverified from anything present here,
and D0's new derivation permission cannot be applied to NAF until its actual
licence is read at the source. Two consequences: the licence check is a Phase 0
task, not a Phase 4 formality, and `natten`'s platform problem is *NAF's*
dependency rather than Pixal3D's.

**4. Pixal3D's own backend selectors default to the two things unavailable here.**
`docs/ideas/Pixal3D/pixal3d/modules/sparse/config.py:3-5` sets
`CONV = 'flex_gemm'` and `ATTN = 'flash_attn'`. Two details matter beyond that:

- `ATTN = 'sdpa'` **is** implemented in Pixal3D
  (`pixal3d/modules/sparse/attention/full_attn.py:232`), and the selector list
  also carries `flash_attn_4`. Pixal3D's base is therefore *newer* than the
  vanilla upstream in `docs/ref/TRELLIS-2/`, and it already contains the exact
  portability fallback the ROCm fork had to add. `backend-parity` D2b's
  provenance table attributes `sdpa` to the ROCm fork alone; that is now known to
  be incomplete, and the table should say so.
- `CONV = 'none'` is *allowed* by the selector list (`config.py:20`) and
  `conv_none.py` is **absent** from `pixal3d/modules/sparse/conv/`. The same empty
  slot D2b describes for vanilla, in a third package. A Pixal3D reference run
  needs a real sparse-conv substitute supplied deliberately, and the choice must
  be recorded as *what ran*, not as what the banner printed.

**5. Numeric dumps run on the CPU, and that sets the tier.** `backend-parity` D2
addendum and `rocm-native-reference` D1 addendum: a ROCm-produced reference made
`test_slat` fail `out7` at rel-L2 1.103 while the port was correct; the same taps
are 4.3e-07 against a `--device cpu` dump. Every numeric fixture this plan gates
on is therefore CPU-produced. The cost is real — the SLAT dump went from ~4 min
to ~75 — and Pixal3D defaults to `1536_cascade`. Phase 0 pins the tier at 512
first, then 1024, and budgets hours.

**6. The 1024 defect named in Update point 3 above is fixed, which changes what
this plan can be measured against.** Query-blocked exact attention moves the
finest-level expansion at 1024 from 3.55 to 4.06 with *fewer* non-manifold edges
(`docs/progress/backend-parity_1024-exact-attention.md`). Point 3's warning stands
as written — `proj` conditioning does not change which kernel runs — but its
consequence does not: the confound is now removable, so "a mesh visibly closer to
the input image" is measurable for the first time. It costs 2.3x end to end, on
both sides of the A/B. See D9.

**7. The NAF acceptance gate is unreachable as written, for a reason that is
upstream of NAF.** The criterion asks for `hr_features` "within the DINOv3 gate
(rel-L2 ≤ 1e-5)". `docs/VERIFICATION.md:43` documents ≤ 7e-7, but against the
reference regenerated in `rocm-native-reference` phase 1 — verified correct to
float64 — the port measures `embd` **4.618e-04** and `cond` **1.623e-04**,
identical on CPU and GPU, `ggml_conv_2d`'s F16 im2col suspected and unconfirmed.

Under `cross` that is tolerable: 1029 tokens enter as a permutation-invariant K/V
set and an error of that size diffuses. Under `proj` the same patch map is sampled
**pointwise** at projected pixel positions, fed through NAF and added per block —
so the port's least accurate stage becomes the geometry conditioning's input. The
open `dino` item therefore changes category, from a known inaccuracy to a
precondition. Three resolutions, and the plan must pick one: gate NAF against the
*reference's* DINO features and accept that the end-to-end chain is then
ungated; move the gate to 2e-04 and say why; or fix the im2col precision first.
Only the third leaves the plan's fidelity claim intact.

**8. Smaller corrections.** `T2_CAPI_ABI_VERSION` is at **19**
(`src/trellis2_capi.h:33`), so Phase 6 lands 19 → 20 — the criterion said 17 → 18
and Phase 6 said 15 → 16, which already disagreed. FOV is now a **field on
`t2_gen_options`**, not a new `t2_generate` argument: that function had 15
arguments, exactly purego's limit, which is why the struct exists (ABI 18 → 19).
Adding a field is cheap and cannot break the Go host at run time. And the DINOv3
checkpoint is the same one this port already converts —
`dinov3-vitl16-pretrain-lvd1689m`, Pixal3D via the `camenduru` mirror
(`inference.py:28`) — so `ggufs/dino_*` should be reusable in the manner of D8,
subject to confirming the mirror is byte-identical to `facebook/`'s. Note that
Pixal3D pins `transformers==4.57.3` against the 4.57.1 installed here: same minor,
so the D2c branch is not taken on either, and the reference is consistent with the
existing DINO fixtures. That holds only until someone upgrades to 5.x.

## Update 2026-08-18b — NAF's source arrived, and it refutes D4

`docs/ref/NAF/` was supplied after the section above was written, which closes
the "not on disk" half of point 3 and opens four questions that could not be
asked before. One of them invalidates an architecture decision.

**1. Apache-2.0, and the weights are on disk.** `docs/ref/NAF/LICENSE` is
Apache License 2.0 and the repository ships **no `NOTICE` file**, so §4(d) does
not bind; retaining the licence text and attributing the source discharges it.
`models/NAF/naf_release.pth` is present locally, 2,664,431 B — the size this plan
already carried, so no `torch.hub` fetch is needed at reference time and Phase 4
converts a file rather than a download.

One file inside that repository carries a different header, and it is worth a
line because it is the only one: `src/layers/rope.py` opens with *"Copyright (c)
Meta Platforms, Inc. … may be used and distributed in accordance with the terms
of the DINOv3 License Agreement"*. It is the **only** file under `src/` or
`utils/` with any copyright header at all, and Valeo shipping their copy under
Apache-2.0 does not relicense Meta's code. So D0's derivation permission does not
reach that one file.

This changes no work. D5 already says the RoPE is built from this port's own
DINOv3 implementation, the project already carries the DINOv3 License for the
DINOv3 *weights* (`THIRD_PARTY.md` §4), and nothing here needs Meta's source
text. It is recorded only so that a later file-by-file transcription under D0
knows which single file to skip — `backend-parity` D2b one level further down: a
repository licence is not a file licence.

**2. D4's exactness argument is false, and the cause is `GroupNorm`.** D4 stated
that NAF's parts "are all local", so evaluating it at the projected sample
positions plus a halo is exact. `docs/ref/NAF/src/layers/convolutions.py:34,42`
puts `nn.GroupNorm(num_groups=8)` in every `EncBlock`, twice, and GroupNorm
reduces mean and variance over **the whole spatial extent** of each group. There
is no halo that makes that exact — every output pixel depends on every input
pixel through the group statistics. A sparse image encoder would need the full
map's statistics anyway, at which point it has done the dense work.

D4 is rewritten below rather than deleted, because the useful half survives: the
*attention* is local and can be evaluated per sample position. The split is
encoder dense, attention sparse.

**3. What NAF actually is.** The plan's "2-layer conv image encoder, an adaptive
average pool of the LR features, and a 9×9 neighborhood attention" is wrong in
three places. From `docs/ref/NAF/src/model/naf.py`:

- **Two parallel image branches**, not one: `encoder` (`kernel_size=1`,
  `ks_res=1`) and `sem_encoder` (`kernel_size=3`, `ks_res=3`), each a `Conv2d`
  followed by `img_layers=2` `EncBlock`s of norm→SiLU→conv→norm→SiLU→conv with
  reflect padding, concatenated to `dim=256`. Five convolutions per branch, not
  two layers.
- **The adaptive pool is on the image path, not the features.**
  `adaptive_avg_pool2d` takes the encoded *image* to `output_size`; the LR
  features are never pooled. `KeyEncoder` pools that same map down to the feature
  grid to make the keys, and **`values = features`** — the raw DINOv3 patch map,
  through no encoder at all.
- **RoPE is applied to the image map only** (`ImageEncoder.forward`), so to
  queries and, through the pooling, to keys. Not to values.

**4. The 9×9 is dilated, which makes it a plain LR neighbourhood.**
`CrossAttention.forward` upsamples k and v to the query resolution with
`nearest-exact` and then sets `dilation = (hq // hk, wq // wk)`
(`src/layers/attentions.py:55`). Dilation equal to the upsample ratio means the
81 taps land on 81 **distinct LR cells** — so despite running at output
resolution, the operation is a 9×9 neighbourhood on the LR feature grid with no
K/V interpolation anywhere. Simpler than D4's "four bilinear neighbours and a
convolution halo", and it holds only while the ratio divides exactly: 512/64 = 8
and 1024/64 = 16 in the three Pixal3D configs, both clean, with `//` silently
mis-dilating if that ever stops being true.

**5. Two risks this kills, one it adds.** The internal bilinear pre-downsample —
the plan's "part most likely to be got wrong" — fires only when the guide exceeds
**4×** the target (`src/model/naf.py:39`). Pixal3D's configs are 512→512,
1024→512 and 1024→1024, so it **never fires**, and the adaptive pool is either
identity or a clean 2×2 mean in all three. The RoPE question is answered too:
`rope_base=100.0`, `normalize_coords="separate"`,
`periods = base^(2i/(D_head/2))` — the DINOv3 axial formulation verbatim, and
`rope_rescale=2.0` sits behind `if self.training`, so it is inert at inference.

The new one: **NAF's guide is the *unnormalised* image in [0, 1]**
(`image_conditioned_proj.py:504` clones it before the ImageNet `Normalize`),
while the DINO path sees the normalised tensor. Two different inputs from one
image, and nothing errors if they are swapped.

**6. The reference materialises `hr_features` densely.**
`image_conditioned_proj.py:541-553` runs NAF at `naf_target_size` over the whole
map and *then* samples it with `proj_grid`. So "dense NAF, then sample" is the
reference's own structure and needs no exactness argument; a sparse attention is
an optimisation measured against it, which is the right way round for the
acceptance criterion.

**7. `natten`'s two code paths, and which one to reimplement.** The modern branch
is pinned to `backend="cutlass-fna"` (`src/layers/attentions.py:71`) — CUTLASS,
i.e. CUDA, so it cannot run here at all. The legacy branch is `na2d_qk` → scale →
`softmax(-1)` → `na2d_av` (`:16-29`), which is exactly gather → scale → softmax →
weighted sum over the dilated window. That is the shape to reproduce, and
`natten`'s **border** rule for windows that would leave the grid remains the one
detail to read out of `natten` rather than infer.

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
  tier is a TRELLIS.2 feature, independent of projection conditioning.
  It has its own plan, `docs/plan/1536-cascade.md`. This plan targets the
  1024 tier and inherits 1536 once that one lands.

  Revised 2026-08-18: **on ROCm the tier is fine, and the answer is the
  same one 1024 needed.** The reviewer reports 1536 behaving like 1024 with
  exact attention selected, on images with enough detail to justify the
  tier. The recorded runaway — level growth 4.08 -> 4.21 -> 4.24 ->
  **4.51** at 9.87 % non-manifold — was measured on the **flash** path and
  is the same F16-accumulation defect the 1024 investigation named, one
  tier up; it is not a property of 1536. Mechanically that is what one
  should expect, since the tier only raises the token count and the token
  count is what selected the broken kernel. Two caveats stay: the cost at
  49,152 tokens is higher than the 2.3x measured at 1024 and has not been
  timed, and **Vulkan is a separate question** — it collapses at 1536
  (2.820 expansion, 163,318 triangles over 2,183,769 vertices, 16.06 %
  boundary) and `backend-parity` D6 measured that Vulkan does *not* carry
  the flash defect, so that failure has no explanation yet and exact
  attention is not expected to fix it.

  For this plan the practical consequence is the opposite of what this
  entry said an hour earlier: Pixal3D's `1536_cascade` default is not a
  blocker on ROCm, and the tier stays a non-goal here for the original
  reason — it is orthogonal to projection conditioning and has its own
  plan — rather than because it is broken.

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
| Extra models | — | **NAF** feature upsampler (Apache-2.0, 2.7 MB, `valeoai/NAF`; licence confirmed 2026-08-18 from `docs/ref/NAF/LICENSE`) |

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
      Against a **CPU-produced** reference, with the weight precision and
      the attention mode named — the same forward measures 5.198e-02 on
      ROCm f16/flash and 3.050e-06 on ROCm f32/exact, so an unlabelled
      number says nothing (`backend-parity` D6, D9).
- [ ] The back-projection (`ProjGrid` equivalent) reproduces the
      reference `image_points` / `valid_mask` for the SS, shape and
      texture grid resolutions — this is pure geometry and should be
      near-exact, not merely within tolerance.
- [ ] The NAF upsampler reproduces the reference `hr_features` at the
      sampled positions within the DINOv3 gate (rel-L2 ≤ 1e-5), and the
      sparse evaluation of D4 is shown to be *exact* against a full
      dense NAF pass, not an approximation. **The 1e-5 figure is not
      currently reachable end to end** — the port's own DINO stage sits at
      1.623e-04 on `cond` against a float64-verified reference, and under
      `proj` that tensor is sampled pointwise rather than diffused through
      1029 cross-attention tokens. Either this criterion measures NAF in
      isolation, against the reference's DINO features, or it waits on the
      `ggml_conv_2d` im2col precision. Decide which; do not quietly
      loosen the number (Update point 7). The "exact against a dense
      pass" half of this criterion now applies to the **attention only**:
      the image encoder is not sparsely evaluable at all, because its
      `GroupNorm` statistics are global (Update 2026-08-18b point 2, D4).
- [ ] A Pixal3D generation runs end to end through `t2_generate` and
      produces a mesh visibly closer to the input image than the
      TRELLIS.2 path on the same seed, with `attention_mode` pinned and
      identical on both sides (D9). Recorded as a side-by-side in
      `docs/progress/`. Judged **by eye in the viewer**, not by counters:
      `backend-parity` D1 and its phase 1c measured that the reference
      implementation itself ranks last on boundary and non-manifold edges,
      so "closer to the image" is a product judgement that no edge count
      answers. The publishing route now exists —
      `scripts/ref_generate.py` → `examples/dual_grid_cli` →
      `scripts/ref_publish_generation.py`, which is how the PyTorch mesh
      got into `generations/` — so the reference side can be produced and
      looked at **before** any C++ is written. Worth doing first: it
      answers "is this worth porting" ahead of Phase 1 rather than after
      Phase 5.
- [ ] Loading the existing TRELLIS.2 GGUFs still selects the `cross`
      path and every existing tap in `docs/VERIFICATION.md` is
      unchanged. This plan must not move a single existing number.
- [ ] `docs/VERIFICATION.md` gains rows for: back-projection, NAF,
      `proj` SS-flow forward, `proj` SLAT-flow forward. The NAF row names
      fp32 weights (D10), which is the same precision as the reference —
      so unlike every other row in that table, it carries no
      weight-precision caveat.
- [ ] `T2_CAPI_ABI_VERSION` bumped and propagated to
      `server/engine.go:20`; verified against the built DLL. The header is
      at **19** as of 2026-08-18 (`t2_gen_options`, `attention_mode`), so
      this lands on 19 → 20 unless something else bumps it first. Read the
      header, do not trust this line — it has been wrong twice.
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
| `docs/ref/NAF/` | the NAF upsampler (Apache-2.0, Valeo) — supplied 2026-08-18; `src/model/naf.py`, `src/layers/attentions.py`, `src/layers/convolutions.py` are the three files Phase 4 reproduces. `src/layers/rope.py` is the one file with a foreign header (DINOv3 License Agreement); use this port's own RoPE instead |
| `models/NAF/naf_release.pth` | the released NAF weights, 2,664,431 B — Phase 4's conversion input, no download needed |
| `docs/ref/TRELLIS-2/` | vanilla TRELLIS.2 upstream (MIT, Microsoft) — the baseline Pixal3D was built from |
| `docs/ref/TRELLIS-2_rocm/` | the ROCm-adapted fork; its diff against vanilla is the gfx1201 workaround checklist (see Update, point 4) |
| `THIRD_PARTY.md`, `docs/VERIFICATION.md`, `server/engine.go` | obligations from the criteria above; `THIRD_PARTY.md` also carries D0's attribution if derivation is chosen |

## Architecture decisions

**D0 — Direct derivation is permitted. Resolved 2026-08-18 by the
reviewer.** This decision was raised on 2026-08-16 as blocking Phase 2
and is now closed.

The clean-room constraint was the second reading: an unstated policy
choice rather than a licence requirement. It came from the generic
project template this repository was started from, where the target is
hard copyleft — GPL/AGPL/LGPL — and the extension to the TRELLIS
reference was collateral. MIT was never the concern. `AGENTS.md` and
`THIRD_PARTY.md` §4 have been corrected accordingly.

So: **MIT source may be translated into this repository's C++ provided
the copyright notice travels with it** — an entry in `THIRD_PARTY.md`
under §1, attribution in the touched files, Microsoft's and Tencent's
notices retained. Apache-2.0 (NAF, dinov2) additionally carries a
`NOTICE` obligation. Weights remain a separate question with separate
terms; a code licence says nothing about them.

Two consequences for the shape of this plan. Phases 1, 3 and 5 shrink
considerably: `ProjGrid`, `ProjectAttention` and the NAF neighbourhood
attention become transcription against a known-good original rather than
reconstruction from observation. And **the parity gates do not change** —
a derived implementation still has to prove it reproduces the reference
numerically, because a translation can be wrong in ways reading cannot
catch. Phase 0 therefore keeps its full weight even though the code may
now be read: the dumps stop being the only channel through which the
reference speaks, but they remain the only channel that *proves*
anything.

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
tokens — worth measuring rather than assuming.

Revised 2026-08-18: this used to expect the `proj` forward to be
measurably **cheaper**, and an adjacent measurement argues against it.
Switching only the *cross*-attention from flash to exact cost +7 % end to
end and changed nothing about the result — "only self-attention matters"
(`docs/progress/backend-parity_1024-exact-attention.md`). Cross-attention
is thus a small share of the flow time, so 1029 → 5 tokens saves little,
while `proj_linear` adds a matmul per block and self-attention (23,358
tokens at 1024) is untouched. Expect a wash, and measure it anyway.

**D3 — Project only the active voxels.** The reference builds the dense
R³ proj grid and then indexes it at `coords`
(`docs/ideas/Pixal3D/pixal3d/pipelines/pixal3d_image_to_3d.py:275-281`).
At the HR stage that is 64³ or 96³ entries for a few tens of thousands
of active voxels. The port projects the active coordinates directly.
Same values, no dense intermediate.

**D4 — Dense image encoder, sparse attention.** Rewritten 2026-08-18
after reading `docs/ref/NAF/`. The original claimed all three parts are
local and that a halo therefore makes a fully sparse evaluation exact.
That is false: `EncBlock` carries `nn.GroupNorm(num_groups=8)` twice
(`src/layers/convolutions.py:34,42`), and those statistics reduce over the
whole spatial map, so no halo makes the encoder exact (Update 2026-08-18b
point 2).

The split that does work follows the model's own shape:

- **Dense**, and cheap: the two image branches → concat to `dim=256` →
  `adaptive_avg_pool2d` to `output_size` → RoPE. 256 channels at 512² or
  1024² is 134 MB or 537 MB in fp16, and it is the only part that needs
  global statistics. `KeyEncoder` then pools that map to the LR grid,
  which is 64² and free.
- **Sparse**, per projected sample position: one query vector from the
  dense map, 81 keys from the LR-grid key map, 81 values from the raw
  DINOv3 patch map, scale, softmax, weighted sum. Exact by construction,
  because every query row of an attention is independent of every other —
  the same property `sdpa_auto`'s query blocking relies on.

What this buys is not the encoder but the **output**: the reference
materialises `hr_features` at 1024 channels over the full target grid
(1.07 GB fp32 at 512², 4.3 GB at 1024²) and then samples it. Evaluating
the attention only where `proj_grid` will sample avoids that allocation,
and that dense 1024-channel map is the one thing worth avoiding.

The 81 taps are a plain 9×9 **LR** neighbourhood rather than an
interpolation: the reference's `dilation = (hq // hk, wq // wk)` equals
the nearest-upsample ratio, so the dilated window hits 81 distinct LR
cells (Update point 4). No bilinear neighbours and no convolution halo —
the two things the original D4 built its argument on.

**D5 — Neighborhood attention as an own op.** `natten` is the
reference's convenience, not an architectural requirement: 9×9 means 81
keys per query, a gather plus softmax over a window dilated by the
LR-to-target ratio. Confirmed 2026-08-18 against `docs/ref/NAF/`, and
`natten` cannot be used here in any case — the path NAF takes with a
current version is pinned to `backend="cutlass-fna"`, which is CUDA. The
legacy path (`na2d_qk` → scale → softmax → `na2d_av`) is the definition
to reproduce.

Two revisions from reading the source. The RoPE question is **answered**:
it is the DINOv3 axial formulation verbatim, `rope_base=100.0`,
`normalize_coords="separate"`, `periods = base^(2i/(D_head/2))`, with
`rope_rescale` inert outside training. And it must nevertheless be built
from **our** DINOv3 RoPE rather than NAF's, because
`docs/ref/NAF/src/layers/rope.py` is under the DINOv3 License Agreement
rather than the repository's Apache-2.0 (Update 2026-08-18b point 1) —
the one file in either reference that D0's permission does not cover.

What still has to be read out of `natten` rather than inferred is the
**border rule** for a window that would leave the grid. Neighborhood
attention shifts such windows inward rather than padding them, but the
exact convention decides 81 taps for every LR cell within four of an
edge, and a wrong rule reproduces plausibly and is wrong everywhere.

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

**D9 — Both sides of every comparison pin the attention mode, and it goes
in the manifest.** Added 2026-08-18. This is `backend-parity` D0 applied
to a second variable, for the same reason: that decision exists because a
512 comparison ran with `background = AUTO` against a `KEEP` reference,
which made the two runs incomparable while every manifest recorded the
mode and nobody read it.

`attention_mode` is now a per-job option (`t2_gen_options`, ABI 19) whose
AUTO default selects flash at HR token counts, and the mode changes
geometry — the finest-level expansion at 1024 moves 3.55 → 4.06 between
flash and blocked exact. A Pixal3D-vs-TRELLIS.2 A/B in which the two
sides differ in that setting measures the kernel and reports it as
conditioning. So: pinned explicitly, identical on both sides, named in
the progress entry, and recorded per generation. The cost is 2.3x end to
end, twice.

**D10 — NAF ships as a single fp32 GGUF.** Added 2026-08-18 on the
reviewer's preference, and the checkpoint makes it the cheap choice rather
than a trade.

`models/NAF/naf_release.pth` contains **only `FloatStorage`** — it is
already fp32 throughout, 2,650,224 B of raw tensor data in 39 tensors. So
`--ftype 0` is a **lossless copy**, while an f16 variant would spend real
precision to save 1.3 MB. There is therefore no f16/f32 pair for this
model: `ggufs/naf_f32.gguf` only, and `scripts/convert_all.sh` keeps it
out of the loop that sweeps both precisions.

The reason to care at all is Update 2026-08-18b point 7's chain. NAF sits
directly downstream of the port's least accurate stage — DINO's patch
embedding at 1.623e-04 — and directly upstream of the geometry
conditioning. Removing an avoidable error source in that chain costs one
megabyte, which is not a trade worth making in the other direction. The
flow DiTs are the opposite case: f32 doubles ~5 GB each and puts the
16 GB tier out of reach, which is why `backend-parity` treats their
precision as a real decision and this one is not.

Two structural facts fall out of the same inspection, and both simplify
Phase 4. Every parameter lives under `image_encoder` — `CrossAttention`
declares no `nn.Parameter`, so **the attention is weightless** and the
whole checkpoint is the two conv branches. And
`image_encoder.rope.periods` is a **stored buffer**, so the 16 RoPE
frequencies are read from the GGUF rather than recomputed from
`base = 100.0`. That makes the licence footnote in Update 2026-08-18b
point 1 moot in practice: the numbers come from the weights file, not from
anybody's source text.

## Plan (phases)

- [ ] **Phase 0a — Reference environment and the fixtures that need no NAF.**
      No container: the uv path from `rocm-native-reference` phase 0
      (`pyproject.toml` extras, `.python-version`, `scripts/ref_env_check.py`,
      `docs/reference-environment.md`), on the **CPU**, per Update points
      1 and 5. Supply Pixal3D's sparse backends deliberately — its
      defaults are `flex_gemm`/`flash_attn` and neither exists here
      (Update point 4) — and record which module actually ran, not which
      selector was set. Write `scripts/dump_pixal3d_reference.py`
      producing taps for `ProjGrid` output (`image_points`, `depth`,
      `valid_mask`, sampled `z_proj_lr`), `z_global`, and one `proj`
      forward per DiT, at 512 first. The SS stage takes no NAF
      (`docs/ideas/Pixal3D/inference.py:26-33` — only `shape_512`,
      `shape_1024` and `tex_1024` set `use_naf_upsample`), so this is a
      complete, parity-gated fixture set for Phases 1, 2 and 3 that does
      not touch the dependency problem at all.
- [ ] **Phase 0b — NAF's reference dump.** Gates Phase 4 and therefore
      Phase 5. Two of the three items this phase carried are now done:
      the source is on disk (`docs/ref/NAF/`) with the weights beside it
      (`models/NAF/naf_release.pth`, so no `torch.hub` fetch), and the
      licence is read — Apache-2.0, no `NOTICE`, with the single-file
      caveat in Update 2026-08-18b point 1.

      What is left is the neighbourhood attention without CUDA. `natten`
      reaches upstream as a Linux/cu124/cp310 wheel and NAF's current-version
      path is pinned to `backend="cutlass-fna"`, so neither runs here; a
      CPU-only source build on Windows/MSVC is unverified and possibly
      unsupported. The alternative is to reproduce the legacy path —
      `na2d_qk` → scale → softmax → `na2d_av` — as `F.unfold` + softmax in
      the dump script, which is the definition rather than an
      approximation (D5). Read `natten`'s border rule first; that is the
      one part that cannot be inferred from NAF's source.

      Then dump `hr_features` densely at `naf_target_size`, which is what
      the reference itself produces (Update point 6), plus the encoder's
      intermediate map so the dense/sparse split of D4 can be gated
      separately from the attention. Guide image **unnormalised**, in
      [0, 1] (Update point 5).
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
- [ ] **Phase 4 — NAF.** `scripts/convert_naf_to_gguf.py` producing
      `ggufs/naf_f32.gguf` from `models/NAF/naf_release.pth` (2,664,431 B,
      torch zip, 39 tensors, all fp32) — `--ftype 0` only, per D10 — the
      dense image encoder — two branches, five convs
      each, `GroupNorm`, reflect padding, `adaptive_avg_pool2d`, DINOv3
      axial RoPE from our own implementation — and the sparse
      neighbourhood attention over the dilated 9×9 LR window (D4, D5).
      `THIRD_PARTY.md` entry under §1 with the Apache-2.0 notice, and the
      DINOv3-licensed `rope.py` explicitly **not** derived from.
- [ ] **Phase 5 — Shape and texture cascade.** Shape 512 → upsample ×4 →
      quantized HR scaffold → shape HR → texture, with `proj_in = 2048`
      (LR‖HR concat), on the existing 1024 tier. Whether the HR scaffold
      quantization has to follow Pixal3D's rounding rather than
      TRELLIS.2's truncation is decided here (see the open question
      below), because it changes which voxels the HR flow ever sees.
- [ ] **Phase 6 — C API, server, docs.** NAF GGUF path and proj
      capability bit in `t2_pipeline_load`, FOV as a **field on
      `t2_gen_options`** rather than a new `t2_generate` argument (that
      function hit purego's 15-argument limit, which is why the struct
      exists), ABI 19 → 20, `server/engine.go:20`, `README.md`,
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
- ~~**D0 is unresolved and gates Phase 2.**~~ **Resolved 2026-08-18:
  direct MIT derivation, with attribution.** What remains is an
  obligation rather than a question — every derived file carries its
  notice, `THIRD_PARTY.md` gains the entries, and the parity gates are
  unchanged because a transcription can still be wrong.
- **Phase 0 is the real risk, and it is now split so that only half of it
  is risky.** Revised 2026-08-18. There is no container and no CUDA
  device, and `natten` reaches upstream as a Linux/cu124/cp310 wheel
  rather than a source build (Update point 2), so the original phrasing —
  "a reference container in which `natten` builds" — describes something
  that cannot happen here. Phase 0a needs none of it and unblocks Phases
  1, 2 and 3. Phase 0b is where the risk went, and it is larger than
  stated: NAF's source is not in this tree at all, only a `torch.hub`
  reference to `valeoai/NAF` (Update point 3).
- **fp32 storage is inferred** from the published blob sizes, not read
  from the safetensors header. The first converter run settles it.
- ~~**NAF's RoPE may not match the DINOv3 axial formulation** despite the
  shared base of 100.0 (D5).~~ **Answered 2026-08-18: it is the same
  formulation**, from the same Meta source file. The risk that replaced it
  is a licence one — that file is under the DINOv3 License Agreement, not
  the repository's Apache-2.0, so it must be reimplemented from our own
  DINOv3 RoPE rather than derived (Update 2026-08-18b point 1).
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
- ~~**Halo correctness in the sparse NAF evaluation** (D4) — the internal
  bilinear downsample when the guide exceeds 4× the target size is the
  part most likely to be got wrong.~~ **Both halves are settled, one in
  each direction.** The pre-downsample never fires in any Pixal3D config
  (it needs guide > 4× target; the configs are 512→512, 1024→512,
  1024→1024). The halo does not need to be right because there is no halo:
  the encoder runs dense, since `GroupNorm` makes a sparse one inexact at
  any radius (D4).
- **`natten`'s border rule is the remaining unknown in NAF.** Windows within
  four LR cells of an edge cannot be reproduced from NAF's source, only
  from `natten`'s. Every projected sample near the silhouette boundary
  depends on it, which is precisely where the geometry is decided.
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
