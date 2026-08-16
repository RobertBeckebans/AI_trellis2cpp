# `backend-parity` — make the compute backends agree, or say by how much they do not

- **Status:** in progress — phases 1c and 2 done, 0/1/1b/3 open
- **GitHub Issue:** none — tracked by plan key (`backend-parity`), so progress
  entries are `docs/progress/backend-parity_<phase>-<name>.md`.
- **Branch:** `pytorch-comparison` (current) or a fresh one
- **Area:** src / tests / docs
- **Date:** 2026-08-16
- **Author:** AI (prepared for review)

## Goal

CPU, ROCm and Vulkan should produce the same mesh from the same seed and differ
only in speed. They do not. Quantify the gap per stage, close what can be
closed, and put a number on the rest instead of leaving it to the eye.

## Why this is worth a plan

The reviewer's expectation was reasonable and is not met: three backends, three
different meshes from one seed. Until this week that could be waved away as
"chaotic sampler drift" — a phrase `docs/VERIFICATION.md` used and which turned
out to be mostly an artefact of one inaccurate kernel. With exact attention the
GPU sampler reproduces the reference at 1.494e-05 and 100 % sign agreement, so
the chain is not inherently chaotic. Divergence has causes, and they measure.

**The shipping configuration fails parity on Vulkan.** SS-flow forward against
the CPU fp32 golden, tolerance 2e-03:

| | f16 (what the server loads) | f32 |
|---|---|---|
| ROCm | 5.463e-04 | **3.050e-06** |
| Vulkan | **1.769e-02** — 8.8× over the gate | 8.717e-04 |

That is one forward, no sampler, no chaining. Over 24 chained forwards and four
subdivision levels thresholded at zero it becomes a different object.

**Vulkan has two separate problems.** Its floor even at f32 is 8.717e-04 against
ROCm's 3.050e-06 — a factor of 286 that is the kernel itself, most likely f16
accumulation in the `KHR_coopmat` path. On top of that, f16 weights cost it
another 20×. f32 weights would move it from failing to passing without bringing
it near ROCm.

**And the CPU sits between the two GPUs, which rules out the obvious reading.**
Einstein, seed 42, 1024³:

| | CPU (ground truth) | ROCm | Vulkan |
|---|---|---|---|
| vertices | 2,854,472 | 2,838,357 | 2,940,217 |
| triangles | 5,732,416 | 5,689,864 | 5,907,996 |
| boundary edges | **0.1157 %** | 0.0415 % | 0.1716 % |
| non-manifold | **43,204** | 22,814 | 53,593 |
| finest expansion | 4.004 | 3.997 | 4.008 |
| wall clock | 6,038 s | 321 s | 232 s |

Deviation from the CPU: ROCm −0.56 % on vertices but −47 % on non-manifold
edges; Vulkan +3.00 % on vertices and +24 % on non-manifold. **Neither
reproduces the reference, and they miss it in opposite directions.** ROCm's
mesh looks smoother than the CPU's — that is a deviation, not a quality.

**The same three backends at 512 split the problem in two** (measured
2026-08-16, all `background = KEEP`, seed 42, same input file):

| | 512 verts | vs CPU | 1024 verts | vs CPU |
|---|---|---|---|---|
| CPU | 711,670 | — | 2,854,472 | — |
| ROCm | 711,680 | **+0.0014 %** | 2,838,357 | **−0.56 %** |
| Vulkan | 728,910 | +2.42 % | 2,940,217 | +3.00 % |
| PyTorch reference | 920,630 | +29.4 % | 3,674,840 | +28.8 % |

**ROCm and the CPU are ten vertices apart at 512** — visually
indistinguishable, and 400× closer than the same pair at 1024. Vulkan is off by
a near-constant 2.4–3.0 % at both tiers. So the two GPU backends diverge for
different reasons: Vulkan's is a flat accuracy offset consistent with its 286×
worse per-forward floor, while ROCm's appears only in what the 1024 tier adds.
One explanation will not cover both, and Phase 1 should stop looking for one.

What "the 1024 tier adds" is deliberately vague: the cascade brings a scaffold
upsample, a second flow model, the 1024 conditioning, one more decoder
subdivision level, and the row counts at which chunking engages. The
measurement narrows the search to that set. It does not name the culprit.

## Non-goals

- **Making Vulkan the equal of ROCm.** Its floor is a kernel property; this plan
  measures it and may mitigate it with f32 weights, but fixing `coopmat`
  accumulation is upstream work in ggml.
- **The mul_mat column truncation.** Separate defect, separate note
  (`docs/bugs/ggml-rocm-mul-mat-column-limit.md`), already reduced to a
  ten-line PyTorch reproducer in `tools/rocblas_col_truncation.py`. It is
  AMD's, not ours.
- **Deciding which backend ships.** That is a product call once the numbers
  exist. For what it is worth today: ROCm at 1024 and above, Vulkan only for
  fast iteration, and never Vulkan at 1536.

## Architecture decisions

**D0 — Every comparison names its `background` mode.** Added after a wrong
number went into a progress doc: a 512 comparison used runs made with
`background = AUTO` against a reference driver that takes `preprocess_image`'s
alpha branch, i.e. `KEEP`. Different preprocessing is a different conditioning
image, so the two were never comparable, and the error was invisible because
the manifests record the mode but nobody read it. The reference driver is
always KEEP; so must every run measured against it be.

**D1 — Mesh counters are not an accuracy metric.** This is the load-bearing
decision, because it inverts the obvious approach.

ROCm is **286× more accurate per forward** than Vulkan, yet its mesh sits
*further* from the CPU on the edge counters (−47 % non-manifold against Vulkan's
+24 %). Boundary and non-manifold counts are the outcome of a threshold applied
to millions of voxels: a small shift in the latent moves them by tens of percent
in either direction. They answer "does this mesh look torn", not "is this
backend correct".

Consequence: correctness is judged on **taps**, which are deterministic and
localised. Meshes are judged separately, as a product question, and the two must
not be conflated — including when the prettier mesh is the less faithful one.

**D2 — PyTorch is the reference. The CPU is a proxy, and only where no PyTorch
reference exists.**

The tap measurements already work this way and should stay that way:
`tests/ss_flow_ref.bin`, `dumps/reference_dino.gguf` and the rest are PyTorch
output, so `test_ss_flow_forward` and `test_dino` compare a backend against the
reference implementation directly, not against another backend.

The mesh comparison in this plan does **not**, and that is a gap rather than a
design. It used our own CPU run as the yardstick because that number was on
hand. The CPU earns that role by reproducing PyTorch (`test_ss_flow_forward`
2.416e-04, `test_ss_dec` 6.172e-07, `test_chunked_decode` exactly 0) — but
"reproduces the reference on the taps we measured" is a weaker claim than "is
the reference", and stacking a backend comparison on top of it inherits every
error the CPU path itself carries.

There is a real PyTorch reference for the decoded voxel set:
`scripts/dump_cascade_reference.py` captures `out7` and `out_coords`, and
`test_cascade` section 3 already gates against them. It is unavailable today
only because the dump on disk was written with `--skip-hr-sampler` and therefore
has no `hr_slat` to decode from. Producing the full dump is
`rocm-native-reference` phase 3, and this plan depends on it.

So: taps against PyTorch always. The decoded voxel set against PyTorch as soon
as the full dump exists. The CPU stays useful for the one thing PyTorch cannot
answer — mesh topology after dual-grid extraction, which has no reference dump
at all — and there it is a proxy, labelled as such. "Closest to ROCm" is never a
criterion; ROCm is a candidate, not a reference. CPU runs cost 6,038 s against
321 s, so they are deliberate reference points, not a routine step.

**D3 — A backend-comparison gate has to be tighter than the parity gate.** The
existing 2e-03 tolerance answers "is the port correct" for a single forward.
Vulkan passes it at f32 (8.717e-04) while producing a visibly different mesh, so
that gate cannot certify "the backends agree". Whatever bound is chosen for
backend-vs-backend must be justified by what survives 24 chained forwards, not
inherited from the port gate.

**D4 — Diagnostics must fire in both directions.** The finest-level expansion
warning triggers on `ratio > 4`. Vulkan's 1536 collapse ran at **2.82** — below
four — so nothing was logged for the worst failure observed. A ratio well under
4 means the decoder is losing voxels, which is as diagnostic as adding volume.
`tris/verts` (2.39 healthy, 0.075 collapsed) is a second, cheaper tell that
nothing currently reads.

## Acceptance criteria

- [ ] Every parity tap measured on all three backends **against the PyTorch
      dumps**, in one table, with the weight precision named. A tap that passes
      on one and fails on another is recorded as such rather than averaged away.
- [ ] The decoded voxel set (`out7`, `out_coords`) gated against PyTorch per
      backend, not against our own CPU run.
- [ ] Vulkan's SS-flow forward inside the 2e-03 gate in whatever configuration
      is recommended for it, or the recommendation is "do not use Vulkan" with
      the number that justifies it.
- [ ] The question "do f32 flow weights move Vulkan's mesh toward the CPU"
      answered with a measurement, not an argument.
- [x] The expansion warning fires on both collapse and runaway, verified against
      the two recorded failures (2.82 and 4.51).
- [x] `ctest --test-dir build -C Release -LE model` green throughout. A `SKIP`
      (77) is not a pass. — 10/10 passed, 0 skipped, as of phases 1c/2. Needs
      the ROCm runtime on `PATH` on Windows or the tests hang in the loader;
      see `AGENTS.md`.

## Context / affected files

| Path | Why relevant |
|---|---|
| `src/trellis2.cpp` | `sdpa_auto` gating, `init_best_backend` device selection, the expansion warning |
| `tests/test_ss_flow_forward.cpp`, `test_dino.cpp` | the taps that already separate the backends |
| `tests/test_chunked_decode.cpp` | asserts bit-identity, which holds on CPU and cannot on HIP |
| `tools/gguf_precision_ab.sh` | end-to-end A/B harness, currently Windows-only |
| `docs/VERIFICATION.md` | the parity table and its platform caveat |
| `docs/plan/rocm-native-reference.md` | where these findings were first recorded |
| `docs/bugs/ggml-rocm-mul-mat-column-limit.md` | the separate rocBLAS defect |

## Plan (phases)

- [ ] **Phase 0 — f32 flows on Vulkan.** The one experiment that follows
      directly from the 2×2 above. `slat_flow_f32` and `slat_flow_1024_f32`
      (5.2 GB each, they exist), decoder left at f16 — because f32 in the
      *decoder* was already measured and changed nothing (2.6321 % → 2.6273 %
      boundary edges), while the *flow* is where the trajectory is set and where
      the error is measured. Compare against tonight's CPU run at seed 42.
      **Deliverable: does closing the tap error close the mesh gap, yes or no.**

      Better motivated since the 512 measurement: Vulkan's offset is
      near-constant across tiers, which is exactly the signature of a kernel
      accuracy floor rather than something the cascade introduces. This is the
      right experiment for Vulkan, and only for Vulkan.
- [ ] **Phase 0b — What does the cascade do to ROCm?** New, and now the more
      interesting half. ROCm reproduces the CPU at 512 to ten vertices and
      misses it by 16,115 at 1024, so its divergence lives entirely in what the
      1024 tier adds. Bisect that set rather than the whole pipeline: run
      `1024` (single-shot, no cascade) against `1024_cascade` on the same seed,
      and run the cascade with `TRELLIS2_NO_CHUNK=1` where it still fits. Each
      of those isolates one suspect — the second flow model, the HR
      conditioning, the extra subdivision level, the chunk boundaries. Cheap:
      ROCm at 1024 is ~320 s, so the whole sweep is under an hour, against
      100 minutes for a single CPU reference point.
- [ ] **Phase 1 — The full tap table, against PyTorch.** Run the suite per
      backend with the runtime switch (`TRELLIS2_DEVICE=cpu|rocm|vulkan`, one
      build). `dino`, `ss_flow_forward`, `ss_sample`, `ss_dec`, `slat`,
      `cascade`. Both weight precisions where affordable. Every one of these
      compares against a PyTorch dump, so the table says how far each backend is
      from the reference — not from each other. This replaces the current
      handful of spot measurements with something citable.
- [ ] **Phase 1b — The decoded voxel set against PyTorch.** Requires the full
      cascade dump (`rocm-native-reference` phase 3): the one on disk was
      written with `--skip-hr-sampler` and carries no `hr_slat`, so
      `test_cascade` section 3 cannot run and `out7` / `out_coords` go
      unchecked. With it, the voxel set each backend decodes is gated against
      PyTorch directly — which is the mesh question asked properly, one stage
      before the dual-grid extraction blurs it into edge counters. Costs one
      full dump: a 1024-model sampler run over ~11k tokens.
- [x] **Phase 1c — A PyTorch generation in the viewer.** Done —
      `docs/progress/backend-parity_1c-2-pytorch-mesh-and-two-sided-diagnostics.md`.
      Einstein, seed 42, type `512`: 920,630 verts / 1,896,504 tris, 0.98 %
      boundary edges, 436 s, published as `generations/6daad1f22a18bbd0`. Route
      taken: `scripts/ref_generate.py` (the upstream stages, stopping at the
      decoder's 7 channels) → `examples/dual_grid_cli` (our extractor) →
      `scripts/ref_publish_generation.py`. The upstream package now sits at
      `trellis2/` in the repo root, gitignored.

      Also done at 1024 (`generations/73f701b208b57c39`, 3,674,840 verts,
      1,716 s), which makes the mesh counters directly comparable: the
      reference has **more** boundary edges (0.2728 %) and **more**
      non-manifold edges (72,713) than CPU, ROCm and Vulkan alike. D1 as a
      measurement rather than an argument — rank by edge counts and the
      reference implementation comes last.

      First number out of it: the reference carries **29.4 % more vertices**
      than our CPU run at 512 (920,630 vs 711,670) and +28.8 % at 1024. The two
      reference runs share a scaffold, so that consistency is not two
      independent samples. Not yet a defect —
      the seed does not transfer between our RNG and PyTorch's — but the size
      of it is what Phases 1 and 1b now exist to explain. The 1024 route is
      also no longer an unknown: the 512 decode took 210 s at 920k voxels, so
      three million is tens of minutes, not hours.

      <details><summary>original phase text</summary>

      The comparison the
      reviewer actually wants: a mesh produced by the reference implementation,
      sitting under `generations/` beside the ROCm, Vulkan and CPU runs of the
      same seed, judged by eye in the same viewer rather than through counters.

      What it takes: the reference chain rerun on the Einstein image (DINO dump
      → `ref_ss_sample.py` → `dump_cascade_reference.py` **without**
      `--skip-hr-sampler`, so `hr_slat` / `out7` / `out_coords` exist), then
      dual-grid extraction from `out7`, then a `.t2mesh` plus a
      `manifest.json` the server picks up at startup.

      Two things are missing. `scripts/ref_dual_grid.py` is referenced by a
      comment in `ref_common.py` but **does not exist**, so there is no Python
      path from decoder output to mesh; and nothing writes a persisted
      generation from outside the server. The `T2MESH01` layout itself is
      already parsed in Python — `load_t2mesh()` in `scripts/ref_texture.py` and
      `dump_texture_reference.py` — so only the writing direction is new, which
      is a short function against a documented format rather than a format
      investigation. The extraction is best done by handing
      the PyTorch voxels to the existing C++ extractor — it is deterministic and
      shared across backends, so any remaining difference is genuinely the
      network rather than the mesher. The artefact is then honestly labelled
      "PyTorch geometry, our extraction".

      **Do it at 512 first.** The reference 1024³ decode runs on the CPU through
      the pure-PyTorch sparse conv (gather + GEMM over 27 kernel offsets, which
      `ref_common.py` itself calls "slow, but it runs anywhere"). At 2.9M voxels
      that is an unknown that has never been run and could be hours. At 512 it
      is ~700k voxels and `dump_slat_reference.py` already produces the chain
      including the decoder output. That proves out the extraction and manifest
      work, and measures what 1024 would cost.

      </details>

- [x] **Phase 2 — Two-sided diagnostics.** Done. The finest-level expansion
      check in `src/trellis2.cpp` now also warns below 3.5 (the Vulkan 1536
      collapse ran at 2.82 and logged nothing), and `src/trellis2_capi.cpp`
      warns when `tris/verts` falls below 1 (that run: 0.075) — ungated, not
      behind `TRELLIS2_TIMING`. Both are print-only.

      Caveat found while doing Phase 1c: the PyTorch reference's own 512 run
      expands 4.484 at the finest level, above the 4.2 upper threshold, with a
      perfectly healthy mesh. The upper bound was calibrated on 1024 runs and
      is a known false positive at 512; left as is, since it only warns.
- [ ] **Phase 3 — The backend-comparison gate (optional).** D3. Only worth doing
      once Phase 1 says what the achievable spread actually is; a bound invented
      before the data would be arbitrary.

## Tests / verification

- Taps against PyTorch, not meshes, for every correctness claim (D1, D2).
- The CPU run of 2026-08-15 (Einstein, seed 42, 1024³, 6,038 s) is the yardstick
  for Phase 0 **only because no PyTorch mesh reference exists** — the reference
  dumps stop at the decoder's 7-channel output, before dual-grid extraction. It
  is a proxy and every number derived from it carries that label.
- `ctest -LE model` green throughout.

## Risks / open questions

- **Whether ggml-vulkan exposes a precision control at all.** The flash-attention
  investigation found `ggml_flash_attn_ext_set_prec()` is silently ignored by the
  CUDA/HIP backend — nothing under `ggml-cuda` reads `GGML_PREC_F32` for that op.
  Whether `coopmat` matmul has an equivalent knob, and whether ggml honours it,
  is **unchecked**. If not, Vulkan's floor is not addressable from this side.
- **f32 flows double the weight footprint.** ~13.5 GB resident for the geometry
  path. Fine on 32 GB, out of reach on 16 GB, which is a tier this project
  otherwise supports.
- **n = 1 on every mesh comparison so far.** The A/B ran one image at one seed.
  A conclusion of the form "backend X is closer" needs two or three seeds before
  it means anything; the `all32` variant already produced a 41 % difference that
  turned out to be a different sample rather than a better one.
- **Vulkan at 1536 is not merely inaccurate, it collapses** (2.82 expansion,
  163,318 triangles over 2,183,769 vertices, 16.06 % boundary edges) and the
  chunking was active, so it is not the mul_mat ceiling. There is a second
  Vulkan limit — it asserted on `maxComputeWorkGroupCount` under
  `TRELLIS2_NO_CHUNK` — and evidently not every op checks it. Out of scope here,
  but it means Phase 1 should not expect Vulkan numbers above 1024 to mean
  anything.
- **The CPU is slow enough to distort the plan.** Every ground-truth point costs
  100 minutes. Budget them deliberately.

## Release note

None user-facing yet. Developer-facing: the compute backend is selectable at
runtime from the viewer and recorded per generation, so two meshes from one seed
no longer differ for a reason nothing preserves.

## Proposed commit message

```text
Plan closing the gap between the compute backends

Three backends produce three meshes from one seed, and the shipping f16
configuration fails parity on Vulkan outright: 1.769e-02 on the SS-flow forward
against a 2e-03 gate, where ROCm sits at 5.463e-04. f32 weights move Vulkan
inside the gate but not near ROCm, whose floor is 286x lower - two separate
problems, one of them the coopmat kernel itself.

The plan turns on one decision: mesh counters are not an accuracy metric. ROCm
is 286x more accurate per forward yet lands further from the CPU on edge counts,
because those counts are a threshold applied to millions of voxels. Correctness
is judged on taps; meshes are a product question, including when the prettier
mesh is the less faithful one - the CPU reference has nearly twice ROCm's
non-manifold edges.

Plan only - no code.
```
