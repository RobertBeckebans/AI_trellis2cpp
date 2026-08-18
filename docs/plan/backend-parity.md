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

Both cells above are the **exact** attention path, which is what SS-flow takes by
default — its score matrix is ~0.8 GiB and fits the gate. Pin the path the other
way and the picture changes completely for one backend and barely at all for the
other; that measurement is D6, and it is what says the two backends need
different recommendations rather than one.

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

**Update 2026-08-18 — it is named, and it is fixed.** What the 1024 tier adds is
*token count*, and the token count decides which attention runs. At 23,358 tokens
and 12 heads the `[L_k, L_q, H]` score matrix is 26 GB in one allocation, so
`sdpa_auto`'s 1 GiB gate always chose flash — whose HIP kernels accumulate in F16
and ignore `GGML_PREC_F32`. Building the exact path in blocks of query rows
removes the memory wall (the split is exact with no correction term, because
`soft_max` runs along keys and every query row is independent), and the finest-level
expansion ratio at 1024 moves from **3.55 to 4.06**, into the healthy band, with
7.9M voxels against 4.3M and *fewer* non-manifold edges. Chunking was never
implicated; it is the attention. One fixture so far. Detail in
[`docs/progress/backend-parity_1024-exact-attention.md`](../progress/backend-parity_1024-exact-attention.md).

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

**D2 addendum, 2026-08-17 — "PyTorch is the reference" needs a device, and it is
the CPU.** The decision above says which *implementation* is authoritative and
says nothing about where it runs. That gap cost two days.
`rocm-native-reference` phase 3 produced the SLAT dump on ROCm, and
`tests/test_slat` failed `out7` at rel-L2 1.103 — in the port, apparently. The
same taps come out at 4.3e-07 against a dump from the same script with
`--device cpu`, 18 of 18 green. The reference's own convolution is wrong on
ROCm; the port never was. Detail in
`docs/progress/rocm-native-reference_3-slat-dump-and-out7.md`.

Two consequences for this plan. **Every numeric reference this plan gates on is
produced on the CPU** — phase 1's tap table and phase 1b's voxel set included.
And a reference disagreement is no longer read as a port defect by default: the
cheap first move is to reproduce the reference's own step from its own captured
tensors, which is what finally separated the two here.

**D2b — which implementation a backend name actually selects.** "PyTorch" is
not always upstream PyTorch, and the selector strings in
`trellis2/modules/sparse/config.py` have accumulated from three sources. A tap
means nothing until it is clear which of these produced it, so the provenance
is recorded here rather than rediscovered:

| Selector | Vanilla (Microsoft) | ROCm fork | Apple port |
|---|---|---|---|
| `spconv`, `torchsparse`, `flex_gemm` | allowed, module shipped | — | — |
| **`none`** | **allowed, no module shipped** | adds `conv_none.py` | — |
| **`pytorch`** | — | — | adds selector + `conv_pytorch.py` |
| `xformers`, `flash_attn`, `flash_attn_3` | allowed, module shipped | — | — |
| **`sdpa`** | — | **adds selector** (`SDPBackend.MATH`) | inherited |
| `flex_gemm_sparse_attn` | — | — | adds selector |

Three things follow.

**`none` is an empty slot upstream.** Microsoft allows the name and ships no
`conv_none.py`, so selecting it in vanilla dies in
`importlib.import_module('..conv_none')`. Two parties fill that slot
independently and differently: the ROCm fork by shipping the file, and
`scripts/ref_common.py` by assigning `conv_dispatch._backends["none"]` directly.
The assignment **bypasses the import**, so it wins — the dispatcher finds the
entry already cached and never loads the file. That is why the dumps this plan
gates on report `Conv backend: none` while running neither Microsoft's code nor
the ROCm fork's, but `ref_common`'s own ~35 lines. The name was right and the
module behind it was not, which cost most of a day.

**`sdpa` and `pytorch` are both portability fallbacks Microsoft does not need.**
Upstream targets CUDA with `flex_gemm` and `flash_attn`; each fork had to replace
one CUDA dependency with something that runs anywhere — `sdpa` swaps flash
attention for the exact materialized softmax (the ROCm fork's answer to F16
accumulation on gfx1201), `pytorch` swaps the Triton sparse convolution for
plain PyTorch (the Apple port's answer to Metal having no Triton). That is the
same move this port makes with ggml, one layer up, and it is why both forks are
worth reading when a kernel here misbehaves.

**Reading a `[SPARSE]` banner is not enough.** It prints the selector, not the
module. When provenance matters, check whether `_backends` was patched.

**The table is incomplete on one row, found 2026-08-18 in a third package.**
Pixal3D (`docs/ideas/Pixal3D/`, built on a *newer* TRELLIS.2 base than the vanilla
copy on disk) already ships `sdpa`: the selector is in its accepted list
(`pixal3d/modules/sparse/config.py:24`, alongside a `flash_attn_4` this project
has not seen) and it is implemented at
`pixal3d/modules/sparse/attention/full_attn.py:232`. So `sdpa` is not the ROCm
fork's invention — either upstream adopted it later or both answered the same need
independently, and the table above cannot tell which. Pixal3D also *allows*
`CONV = 'none'` while shipping no `conv_none.py`, which is the vanilla empty slot
again, in a third package. The lesson is the one this decision already teaches,
one level up: a fork's diff against **the copy you happen to have** is not its
diff against upstream.

**D2c — a library *version* selects an implementation too, and one already does.**
Added 2026-08-18 from the Apple-port comparison. D2b is about a backend name not
naming a module; this is the same failure one layer up, and it is live rather
than historical.

`trellis2/modules/image_feature_extractor.py:97` branches on
`hasattr(self.model, 'model')`. Not taken, it iterates the transformer layers by
hand. Taken, it returns `last_hidden_state` — which is `self.norm(hidden_states)`,
a LayerNorm **with** affine parameters (`modeling_dinov3_vit.py:529`) — and then
applies a *second*, affine-free `F.layer_norm` over it. The manual path never
touches `self.norm` at all. Two different feature tensors, no error either way.

The installed **transformers 4.57.1** does not take the branch (`DINOv3ViTModel`
has `.layer` at the top level and no `.model`), so `dumps/reference_dino.gguf`
and `test_dino`'s 7e-7 are honest today. A 5.x upgrade takes it, moves every DINO
reference, and turns `test_dino` red — pointing at the port. Exactly the shape of
the `out7` episode, and the reason that one cost two days.

Consequence: a `transformers` upgrade is a **reference-affecting change** and must
be treated as one. Either adopt the Apple port's resolution — it looks for
`.layer` on the model and then on `.model`, so both layouts iterate manually and
no version picks a different mathematical path — or pin the version and record
why. `docs/reference-environment.md` is where that belongs.

The general rule the two decisions share: **provenance is not what the code
selects, it is what actually ran.** A selector string, a patched dispatch table
and a library version can each break that link without saying so.

So: taps against PyTorch always, on the CPU. The decoded voxel set against
PyTorch as soon as the full dump exists. The CPU stays useful for the one thing PyTorch cannot
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

**D5 — The attention path is a per-generation choice, and its default does not
move yet.** Added 2026-08-18 with the finding above.

`sdpa_auto`'s 1 GiB gate answered a *memory* question: exact where the score
matrix fits, flash where it cannot be allocated. Query-blocking has removed that
wall, so the gate now answers nothing it was designed for. What is left is a time
question — exact costs 2.3x end to end at 1024 — and there is exactly one
measurement of the quality it buys, on one fixture at one seed.

So: the mode becomes a per-job option (`t2_gen_options.attention_mode`, exposed in
the server UI) rather than a new default, and the gate stays where it is until two
or three further fixtures say what it should be. A default chosen from one sample
is an extrapolation wearing a measurement's clothes, and this plan exists because
that has already cost the project once.

Two consequences worth stating. **Every parity number in this plan was taken on
the flash path** unless it says otherwise, which for the cascade tiers means it
measured a defect rather than the backend. And an option that changes geometry
belongs in the manifest, not just in the process environment — `attention` is
recorded per job for the same reason `background` is (D0).

**D4 — Diagnostics must fire in both directions.** The finest-level expansion
warning triggers on `ratio > 4`. Vulkan's 1536 collapse ran at **2.82** — below
four — so nothing was logged for the worst failure observed. A ratio well under
4 means the decoder is losing voxels, which is as diagnostic as adding volume.
`tris/verts` (2.39 healthy, 0.075 collapsed) is a second, cheaper tell that
nothing currently reads.

**D4 addendum, 2026-08-18 — a two-sided warning still needs its threshold in the
right place.** The lower bound went in at 3.5, chosen against the 2.82 collapse,
and that left far too much room on a metric whose healthy band is 4.00–4.04: a
broken 1024 run measured **3.547 and said nothing**, while a second one at 3.432
warned. Same failure, opposite sides of the line, and the silent one was the run
that started this investigation. It is now 3.8, bracketed by three points (3.432
and 3.547 broken, 4.055 healthy). The general lesson is the one D4 already
implies and did not follow through: a threshold placed relative to the *worst*
observed failure is not placed relative to the healthy band, and it is the healthy
band that defines what "abnormal" means.

**D6 — The two backends fail for opposite reasons, so one recommendation cannot
cover both.** Added 2026-08-18. `test_ss_flow_forward`, one seed, gate 2e-03:

| | f16 flash | f16 exact | f32 flash | f32 exact |
|---|---|---|---|---|
| **ROCm** | 5.198e-02 | 5.463e-04 | 5.198e-02 | **3.050e-06** |
| **Vulkan** | 1.776e-02 | 1.769e-02 | **1.210e-03** | 8.717e-04 |

Read the rows, not the cells. **On ROCm the attention path decides everything and
the weights decide nothing on flash**: f32 and f16 give the identical 5.198e-02,
which is what an error entirely produced by F16 accumulation inside the kernel
looks like. Switch to exact and the weights start mattering again — 3.050e-06
against 5.463e-04, a factor of 180.

**On Vulkan it is the exact opposite.** The attention path is worth almost
nothing (1.776e-02 → 1.769e-02), while f32 weights are worth a factor of 15.
Its flash kernels are a different implementation and evidently do not accumulate
in F16.

Three consequences. The recommendation is per backend, not global: **ROCm wants
exact attention, Vulkan wants f32 weights**, and neither setting helps the other
much. The best configuration measured anywhere is ROCm + f32 + exact at
3.050e-06, and ROCm + f16 + exact (5.463e-04) still beats every Vulkan cell. And
D1's warning applies to this table too — it measures one forward, and the mesh
question needs a generation.

## Acceptance criteria

- [ ] Every parity tap measured on all three backends **against the PyTorch
      dumps**, in one table, with the weight precision named. A tap that passes
      on one and fails on another is recorded as such rather than averaged away.
- [ ] The decoded voxel set (`out7`, `out_coords`) gated against PyTorch per
      backend, not against our own CPU run.
- [x] Vulkan's SS-flow forward inside the 2e-03 gate in whatever configuration
      is recommended for it, or the recommendation is "do not use Vulkan" with
      the number that justifies it. — **Inside, with f32 flow weights**:
      1.210e-03 on flash, 8.717e-04 on exact. With f16 it fails either way
      (1.776e-02 / 1.769e-02). So the recommendation for Vulkan is f32 weights,
      and the attention path is nearly free to choose there. Measured
      2026-08-18, full matrix under D6.
- [~] The question "do f32 flow weights move Vulkan's mesh toward the CPU"
      answered with a measurement, not an argument. — **Half of it is measured.**
      On the forward, f32 weights are worth a factor of 15 to Vulkan
      (1.776e-02 -> 1.210e-03 on flash). Whether that carries through 24 chained
      forwards to the *mesh* still needs a generation, which is what this
      criterion actually asks for.
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
| `scripts/ref_common.py` | produces every reference this plan gates on — and carries its own sparse conv, which is wrong on ROCm (D2 addendum) |
| `trellis2/modules/image_feature_extractor.py` | the version-dependent DINOv3 path (D2c); gitignored working copy |
| `docs/ref/trellis2-apple/` | the second fork the provenance table and D2c were derived from; see `docs/progress/apple-port-comparison.md` |
| `tests/test_slat.cpp` | the tap that exposed it; reports per-channel cos/rms on a failure and dumps a tap with `TRELLIS2_DUMP_TAPS` |
| `src/trellis2.cpp` — `sdpa_auto` | the size gate and the query-blocked exact path (D5); `TRELLIS2_SDPA_BLOCK_MB` sizes the blocks |
| `src/trellis2_capi.h` — `t2_gen_options` | per-job `attention_mode`; ABI 19 |
| `tests/test_ss_flow_forward.cpp` | runs the exact path by default, so a forced multi-block split tests the blocking against an unchanged reference |

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

      **On the CPU** (D2 addendum), which changes the cost: the SLAT dump went
      from ~4 min to ~75 at 24.6 of 32 cores busy, and this one is larger. Most
      of that is the sampler, which the decoded voxel set does not need — a
      variant that loads `hr_slat` from an existing dump and runs only the
      decoder on the CPU would be both cheaper and a cleaner comparison, since
      only one variable moves. Worth building before paying for the full run.
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
- **The cascade numbers in this plan predate the attention fix.** Every ROCm and
  Vulkan figure at 1024 and 1536 above was taken on the flash path, i.e. with the
  defect D5 describes still in it. They are not wrong, but they measure a
  configuration nobody should now run, and the ROCm-vs-CPU gap at 1024 (−0.56 %
  vertices, −47 % non-manifold) in particular needs re-measuring on the exact path
  before anything is concluded from it. The 512 rows are unaffected — that tier
  ran flash too, but at 5,121 tokens it does not move enough voxels to matter.

  **This applies to 1536, and there it has already been overtaken.** The reviewer
  reports that tier behaving like 1024 once exact attention is selected, on images
  with enough detail to warrant it — so the recorded subdivision runaway
  (4.08 → 4.21 → 4.24 → **4.51**, 9.87 % non-manifold) is the flash defect one
  tier up rather than a limit of the tier, which is what the mechanism predicts:
  1536 only raises the token count, and the token count is what chose the kernel.
  Not yet re-measured here, and the cost at 49,152 tokens is untimed. Vulkan's
  1536 collapse is **not** covered by this — D6 measured that its flash kernels do
  not carry the F16 defect, so that failure remains unexplained and the non-goal
  "never Vulkan at 1536" stands.
- **The exact path is measured on one fixture.** Expansion 3.55 → 4.06 on a
  TV-on-a-stand at seed 42, plus `test_ss_flow_forward` at 3.050e-06 across block
  sizes. That is enough to name the mechanism and not enough to move a default;
  two or three further fixtures decide it (D5).
- ~~**Whether Vulkan has the same defect is unknown.**~~ **Measured 2026-08-18:
  it does not.** Pinning the path moves Vulkan by 1.776e-02 -> 1.769e-02 at f16
  and 1.210e-03 -> 8.717e-04 at f32, i.e. almost nothing, where the same switch
  moves ROCm by four orders of magnitude. ggml's Vulkan flash kernels do not
  carry the F16-accumulation defect the HIP ones do. Vulkan's 1024/1536 numbers
  therefore carry one fault, not two — but it is the fault f32 weights address,
  so they still want re-measuring at f32.
- **A `transformers` upgrade silently invalidates every DINO reference** (D2c).
  It is not currently pinned anywhere that a person would look before upgrading.
- **`docs/ref/` is untracked.** The provenance table (D2b), D2c and the
  `conv_pytorch.py` proposal all cite a third-party MIT repository that is not in
  the repository. Either commit it or write down where it came from; the plans
  already carry the same complaint about `docs/ideas/`, which suggests this is a
  habit rather than an oversight.

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
