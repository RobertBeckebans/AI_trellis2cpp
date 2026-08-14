# Progress — 1536-cascade, phase 3: measurement on the R9700

- **Issue:** none — tracked by plan key (`1536-cascade`)
- **Plan:** [`docs/plan/1536-cascade.md`](../plan/1536-cascade.md) — phase 3
- **Branch:** `resolution-1536`
- **Date:** 2026-08-13
- **Commits:** <to be added after review>

## Goal of the phase

Replace the extrapolated numbers from D4 with a real 1536 generation: decode
VRAM peak, voxel count, host RAM high-water, wall clock, and how often the
token-reduction loop actually fires.

## Setup

- Radeon AI PRO R9700, `gfx1201`, 32624 MiB, HIP/ROCm backend
- Go demo server against `libtrellis2.dll`, quality `1536`, PBR on, live
  previews off, 12 shape steps / 12 texture steps
- Two generations from different source images, both requesting 1536

## The headline result

**The token budget did not reduce, in either run.** Both logged

```text
  upsampling scaffold ...
    cascade resolution 1536³
```

so the HR flow ran on the full 96³ scaffold and the decode produced a true
1536³ dual grid. Reduction rate so far: **0 of 2**.

That answers the plan's "diminishing returns" risk in the encouraging
direction — the tier is not quietly a slower 1024 — but two objects is a
sample, not a rate. It also means the *expensive* branch is the one that runs
by default, which is what the timings below are about.

## Run 1 — job `a8ca079175dd94a1`, complete

| Stage | Wall clock |
|---|---|
| loading models | 10.06 s |
| preprocess | 0.03 s |
| encoding image (DINOv3) | 0.38 s |
| sampling sparse structure | 25.43 s |
| decoding occupancy | 4.00 s |
| sampling shape SLAT (LR, 512 model) | 6.19 s |
| upsampling scaffold + token budget | 1.92 s |
| **sampling shape SLAT (HR, 1024 model)** | **122.64 s** |
| **decoding shape (1536³)** | **20.24 s** |
| extracting mesh | 6.52 s |
| **sampling material** | **117.69 s** |
| **total** | **315.12 s** |

- Mesh: **3,321,101 vertices / 6,661,462 triangles**, PBR
- VRAM: never above **16 GB** of the card's 32 GB (external reading)
- Export from the same mesh, for scale: CGAL alpha wrap 60.2 s → 98,622 tris,
  projected 2048² GLB bake 27.7 s (xatlas unwrap 20.9 s of it), 22.9 MiB GLB,
  1.3 % of covered normal-map texels left flat. A second export with quad
  remesh: 34.8 s AutoRemesher → 16,481 quads / 0 boundary edges / 96.8 % area
  kept, 17.3 MiB GLB.

## Run 2 — in progress at the time of writing

Also full 1536³, no reduction. Its HR flow is **far slower**: ~49 s per step
versus ~10.2 s in run 1, so ~4.8×. The LR stages are comparable
(SS flow 26.7 s vs 25.4 s), so this is scaffold density, not machine state —
run 2's object quantizes to many more of the 49,152 allowed tokens. Projected
HR flow alone ~10 min.

Numbers to be completed once it finishes.

## Quality — the RoPE-extrapolation risk

The plan's first risk was that a 96³ scaffold feeds the HR model coordinates up
to 95 while its checkpoint declares `resolution: 64`, and that the failure mode
would be *quality loss that looks like a successful run*.

Run 1's subject was a stylized character (leather jacket, spiked shoulders,
goggles, boots) — a good probe, because it carries fine, semantically obvious
detail that degrades visibly. The result reconstructs the jacket's chest emblem,
the shoulder and sleeve spikes, separate zipper runs, the goggle band, the hair
spikes and the boot soles as distinct geometry.

**Catastrophic RoPE extrapolation is therefore ruled out**: the failure mode
would be mush, noise or structural breakup at exactly those scales, and none of
it is present. The reviewer's assessment of the tier against the lower ones is
"noticeably better".

What this is *not*: a controlled 1024-vs-1536 comparison at identical image and
seed with both results side by side. That remains the plan's stated
verification and is still owed — see Open points.

## Host RAM (D5), from the measured mesh size

Still not instrumented, but no longer unknown. At 3,321,101 vertices and
6,661,462 triangles the `t2_mesh_result` itself is:

| Buffer | Size |
|---|---|
| verts (3×f32) | 38.0 MiB |
| normals (3×f32) | 38.0 MiB |
| tris (3×i32) | 76.2 MiB |
| PBR (6×f32) | 76.0 MiB |
| **total** | **≈ 228 MiB** |

The server copies that once more across the FFI boundary, so the generation
path costs roughly half a gigabyte of host RAM — comfortable, and not the axis
D5 feared. The real consumer is downstream: the CGAL alpha wrap ingests all
6.66M triangles (60 s), and the export chain holds several derived meshes at
once. That path completed without trouble here, but its high-water mark is what
should be sampled if D5 is ever closed properly.

## The character run did not actually test the tier

Worth stating plainly, because it was used as evidence twice and should not
have been:

| run | vertices |
|---|---|
| character at **1536** | 3,321,101 |
| jester at **1024** | 3,440,673 |
| jester at **1536** | 9,715,171 |

The character's 1536 mesh is *smaller* than the jester's 1024 mesh. A sparse,
smooth subject at 1536 lands in the same size regime a dense one reaches at
1024 — so that run never entered the territory where the tier behaves
differently. Its good result says the pipeline is correct at ~3M voxels, which
1024 already showed.

**Every run that actually reached the 1536 regime came out worse than the same
object at 1024.** That is one object, so it is not a rate — but it is the only
evidence about the regime the tier exists for, and it is negative.

## What this changes

**The decode is not the problem.** D4 budgeted the 1536³ decode as the risk
(~15 GB extrapolated) and it turned out to be **20.24 s and comfortably
inside the card** — only ~1.6× the ~12.5 s that `docs/PLAN.md` records for
1024³. The two stages that actually cost are the **HR flow (122.6 s, and
~4.8× that on a dense object)** and the **texture stage (117.7 s)**, neither
of which the plan flagged. Together they are 76 % of run 1.

**`max_num_tokens` must not rise, and may have to fall.** The plan asked
whether the 49,152 budget should rise on a 32 GB card. Two findings say no, and
the second was not anticipated:

- *Time.* A scaffold near the ceiling already spends ~8 minutes in the HR flow.
  Raising the budget raises that with no VRAM benefit to collect.
- *Correctness.* A third object (a jester figure, run 3) failed the HR decode
  outright: decoder level 3 reached 2,151,017 voxels against this backend's
  2,097,152 mul_mat column cap — 2.6 % over. The budget bounds the HR flow's
  **input**; nothing bounds the decoder's **output**, and it is the output that
  meets the ceiling. Plan D4 predicted this in one sentence and it was not
  taken seriously enough: "the surface-scaling assumption ignores that the loop
  caps the input token count but not the decoder's output voxel count."

So the budget is not only a time knob, it is implicitly a decode-size knob, and
49,152 is too generous for this backend's decoder. The safe value depends on
the ratio between scaffold tokens and level-3 voxels, which no run has recorded
— that is what the instrumentation added alongside this entry now reports.

**`T2_PIPE_AUTO` staying at 1024 is vindicated.** A 5-minute best case and a
~14-minute dense case is not a default.

## Deviations / what is still not measured

- **Decode VRAM peak is bounded, not pinned.** "< 16 GB total" is an external
  reading while the flow DiTs and the decoder were still resident
  (`ensure_decode_vram` returns early on this card — free VRAM stays well above
  its threshold, so nothing was freed). The decode's transient share is
  therefore *somewhere below* that ceiling, but the split between resident
  weights and transients was not instrumented. **`decode_vram_peak`'s 1536
  entry is deliberately left at the conservative 15 GB**: the measurement rules
  out that it is too low, which was the dangerous direction, and lowering it on
  an inferred split would trade a safe over-estimate for a guess. Pinning it
  needs a `trellis2_gpu_free_vram()` sample either side of the HR decode.
- **Host RAM high-water (D5) still not instrumented**, only derived from the
  mesh size (see above). The generation path is ~0.5 GB; the export path's peak
  is the open number.
- **Reduction rate** rests on two objects.
- **RoPE quality is argued, not measured.** Catastrophic failure is ruled out
  by inspection (above), but the controlled same-seed 1024-vs-1536 comparison
  has not been recorded.

## Open points

- Finish run 2 and add its stage table.
- Record the same-seed 1024-vs-1536 pair as an artefact, so the quality claim
  rests on something a later reader can re-check rather than on a judgement
  made once.
- Optional: `trellis2_gpu_free_vram()` sampling around the HR decode behind
  `TRELLIS2_TIMING`, to turn the VRAM ceiling into an actual peak and to make
  the host-RAM axis observable.

## Next phase

None planned — phases 1, 2 and 4 landed in
`docs/progress/1536-cascade_phase1-2-4-tier.md`. What remains is the quality
side-by-side and, if wanted, the VRAM/RAM instrumentation.

## Proposed commit message

```text
Record what the 1536 tier actually costs on the R9700
```
