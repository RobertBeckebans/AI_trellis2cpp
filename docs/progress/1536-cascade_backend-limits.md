# Progress — 1536-cascade: the backend limits a dense object runs into

- **Issue:** none — tracked by plan key (`1536-cascade`)
- **Plan:** [`docs/plan/1536-cascade.md`](../plan/1536-cascade.md) — fallout from
  phase 3, not a planned phase of its own
- **Branch:** `resolution-1536`
- **Date:** 2026-08-14
- **Commits:** <to be added after review>

## Goal

Phase 3 measured the tier on one sparse subject and it worked. Every dense
subject then failed, in three different places. This entry is what those
failures were, why the obvious diagnosis was wrong twice, and what the fixes
cost.

## Background: the limit that is *not* the cause

`docs/bugs/ggml-rocm-mul-mat-column-limit.md` documents that `ggml_mul_mat` on
ROCm stops writing past a fixed column count (2^21 for f16/bf16, 2^19 for f32),
silently. The shape decoder already guards against it (`mul_mat_max_rows`,
`shape_dec_final_level_chunked`, `tests/test_large_rows`).

That guard made it the first suspect three times, and it was wrong every time.
It is worth stating why, because the bug note says so itself and it was
overlooked: **only a `[64, 64]` weight was ever measured**, and the ceiling may
be shape-specific. The encoder's convs are `[64, 128]` and `[128, 256]`. Two
runs passed 2^21 at encoder level 0 with correct materials, and forcing the
encoder onto the CPU — where no such ceiling exists — changed nothing. The cap
does not bind at those shapes.

## What actually broke: 32-bit dispatch sizing

A GPU dispatch is sized in **workitems**, in 32 bits. Several kernels build a
grid that scales with the voxel count, so past a few million voxels the launch
is rejected outright (`hipErrorInvalidConfiguration`) and takes the process
with it — the Go server included, so a queued job is lost too.

### A — the encoder's down-sample skip

`ggml_mean` over `[gsz, C_out, Lc]` puts one workgroup on each of `C_out*Lc`
rows with 32 threads apiece (`gsz < 1024`):

| run | `Lc` | `C_out*Lc*32` | outcome |
|---|---|---|---|
| character, 1536 | ~969,000 | 3.97e9 (92 %) | completed |
| jester, 1024 | 858,083 | 3.51e9 (82 %) | completed |
| armchair, 1536 | 1,245,145 | **5.10e9 (119 %)** | **abort** |

Threshold `Lc = 2^20`. Three points, no counterexample — and the character sat
at 92 % of it, so the runs that worked were not comfortably clear of the limit,
they were just under it.

**Fix:** the mean is over `gsz = 4` consecutive channels, so it is now summed as
`gsz` strided slices plus adds and one scale. Same value, out of flat-indexed
copies and broadcasts that carry no per-row dispatch. It stays on the GPU,
which is the point — the objects that trip these limits are exactly the ones
nobody wants to decode on the CPU. A CPU fallback was written first and
discarded for that reason; `TRELLIS2_SHAPE_ENC_CPU` remains as a manual hatch.

### B — decoder level 3 past the column cap, and not splittable

```text
HR shape decode: level 3 has 2151017 voxels, past this backend's mul_mat
column limit (2097152), and only the finest level can be split
```

2.6 % over, after 407 s of HR flow. This one *is* the column cap, and the guard
worked exactly as designed — a readable failure instead of a quarter-lost mesh.

**Why 1536 and not 1024, for the same object.** Level 3 sits three upsamples
above the scaffold, so its voxel count is driven by the *scaffold token count*,
not by the tier's resolution. Quantizing the same candidate cloud onto 64³
merges more voxels than onto 96³, so the 1536 scaffold carries more tokens and
level 3 inherits them.

**Why the token budget did not prevent it.** `max_num_tokens` bounds the HR
flow's **input**; nothing bounds the decoder's **output**, and it is the output
that meets the ceiling. Plan D4 predicted this in one sentence that was not
weighted heavily enough at the time.

### C — the S2C gather, once A was fixed

With A fixed, the jester reached the texture stage with `L = 9,715,171` and
`Lc = 2,147,558`, and aborted anyway: `ggml_get_rows` over `Lc*8 = 17,180,464`
index rows is again just past 2^32 workitems, by about 2 %.

Three limits, all the same shape. **Fixing them one kernel at a time does not
converge, and each attempt costs an 8-minute run to discover the next one.**
That is the finding, not the individual kernels.

## The fixes: split the level, bound every launch at once

### `shape_dec_level_chunked` — an intermediate decoder level

A naive voxel split is wrong: ConvNeXt block *b+1* reads the **neighbours** of
block *b*'s output, which a per-block graph does not have, and with a dozen
blocks a halo would cost more than the level.

The way out is the move the finest level already makes — keep the input
resident over all L and gather from it — applied once per **stage** rather than
once per level:

```text
stage A     y = conv2(silu(LN_free(hch)))            [C, L]
host        h = y + repeat_interleave(xch)
stage B_b   h = convnext_b(h)                        per ConvNeXt block
stage C1    subdiv = to_subdiv(h)                    [8, L]
stage C2    h1 = conv1(silu(LN(h)));  x = h          [C_next*8, L]
```

Every stage reads a full-length resident input, so no neighbour is ever absent.
The per-row ops (norm, silu, affine LN) ride **behind** the gather — that is
what keeps each launch block-sized. Applying them to the resident tensor
instead would rebuild a full-length intermediate and defeat the split; the
first draft did exactly that and had to be corrected.

The child-selection tail is now shared rather than duplicated
(`advance_up_level`). It fixes the next level's voxel order, so two copies of it
would be a divergence nobody finds later. The gather itself stays with each
caller, because the single-graph path reads its conv output in place where the
backend allows it rather than duplicating the full `[C_next*8, L]` — around
8 GB at 1024³.

### `shape_enc_level0_chunked` — the encoder's first level

Same idea, with the twist that this level spans **two voxel spaces**: the convs
and the child gather in fine space (L), the down-sampled result in coarse (Lc).

```text
P0  h    = input_layer(in)                    [C_in, L]      per-row
P1  h1   = conv1(silu(LN(h)))                 [C_out/8, L]   27 fine neighbours
P2  h1c  = s2c(h1)                            [C_out, Lc]    8 children
P3  skip = mean_gsz(s2c(h))                   [C_out, Lc]    8 children
P4  out  = conv2(silu(LN_free(h1c))) + skip   [C_out, Lc]    27 coarse neighbours
```

No per-stage ping-pong is needed here: level 0 reports `blocks = 0`. Only level
0 is eligible and only level 0 needs it — `Lc` falls roughly four-fold per level
and levels 1-3 carry ConvNeXt blocks.

### Encoder levels 1-3 followed

Level 0 was chunked first because it is the largest and the simplest — no
ConvNeXt blocks, so no per-stage ping-pong. That left levels 1-3 exposed, and
at 1536 level 1 does cross the column cap (2,152,267 against 2,097,152) without
being split.

`shape_enc_level_chunked` now covers any level with a down-block: `lvl` selects
the block prefix, level 0 keeps its `input_layer` pass, deeper levels take the
previous level's features unchanged, and the ConvNeXt chain is ping-ponged
exactly as in the decoder — each block reads a full-length resident input,
because block *b+1* needs the neighbours of block *b*'s output.

The gating condition dropped to `has_down && !taps && L > chunk_rows_limit()`.
The diagnostic's `will_chunk` mirrors it, with a comment saying it must: that
predicate has drifted from reality twice, and a warning describing a case that
is in fact handled sends the next investigation somewhere pointless.

## The bug the tests found, then found again

`ggml_gallocr` only gives an input tensor a buffer once it is part of the graph.
The **subdiv head is the one stage of a level that does not convolve**, so its
27 neighbour leaves were created, never entered the graph, and uploading into
them tripped `GGML_ASSERT(buf != NULL)`.

It slipped through because the texture decoder replays a *recorded* subdivision
(`guide`) and never reaches that head — so the first version of the test passed
while a real shape decode died on the card. The fix is an explicit `needs_nbr`
per stage; the lesson is that "untested" was the accurate description and it was
written down one message before it bit.

## Verification

`tests/test_chunked_decode` now runs the same block sweep through **three
models**, because each covers a path the others do not:

| model | covers |
|---|---|
| `tex_dec_f16` | recorded subdivision (`guide`) |
| `shape_dec_f16` | **predicted** subdivision — the subdiv head |
| `shape_enc_f16` | encoder level 0, two voxel spaces |

Block sizes `997`, `333` and `37` on top of the default reach the
**intermediate** levels rather than only the finest. At 37 the fixture's decoder
level 3 splits into 111 blocks and the encoder's level 0 into ~886 fine and ~110
coarse blocks, all with ragged tails.

```text
tex_dec     block=default / 997 / 333 / 37   max|d| = 0 -> OK
shape_dec   block=default / 997 / 333 / 37   max|d| = 0 -> OK   (+ coords compared)
shape_enc   block=default / 997 / 333 / 37   max|d| = 0 -> OK   (+ coords compared)
```

**Bit-identical to the single-graph path in every case**, features *and*
coordinates — the coordinate comparison matters because the subdivision fixes
the next level's voxel order, so matching features on a different coord set
would still be wrong.

Runs locally on CPU with the f16 delivery weights: no GPU, no reference dumps.
`ctest -LE model` → 9/9 throughout; `chunked_decode` → Passed (45 s).

On the card: `test_large_rows 10000000` reports `get_rows`, `add`, `norm`,
`silu` and `repeat` exact at ten million rows, with only the known `mul_mat`
break rows — which is how the "2 GB resident tensor" hypothesis was ruled out.

## Diagnostics added

Each of these exists because a run took 8-16 minutes to say nothing useful:

- `[cascade] requested … achieved … N scaffold tokens …` — the one number that
  predicts the rest of the run, known minutes before anything can fail.
- `[shape_dec] level N: L=… cap=… split block=…` with `<-- PAST THE CAP`.
- `[shape_enc] level N/M: L=… Lc=… blocks=… S2C gather over … rows` with the
  level's own cap and `(chunked)`.
- `[mesh] grid …: … boundary edges …, … non-manifold edges` (see the hole-fill
  bug note).

**Two of these warnings lied before they were corrected**, and both are recorded
because the failure mode is instructive: one kept naming the skip mean after the
skip mean was fixed, the other fired for a model sitting on the CPU where the
cap does not exist. A diagnostic that names a cause which no longer applies is
worse than none — it stops the next search at the wrong place, which is exactly
what it did.

## Settled: the remaining gap is the model, and here is the number

The run that closed it had **every** level chunked — encoder 0, 1 and 2 all
report `(chunked)`, no warning, pure GPU, no environment overrides. It is the
first run with no uncapped matmul anywhere. Same object, same seed 42, same
guidance 7.5:

| | vertices | triangles | boundary edges | **non-manifold edges** |
|---|---|---|---|---|
| 1024 | 3,440,673 | 6,897,372 | 0.171 % | **0.42 %** (43,645) |
| 1536 | 9,715,171 | 22,076,668 | 0.921 % | **7.47 %** (2,369,133) |

**A factor of 17.7 on the non-manifold fraction.** 0.42 % is the floor dual
contouring over a sparse voxel set produces anyway; 7.47 % is a torn surface.
The holes, at 0.9 %, turned out to be the smaller half of the problem.

The same chunked code produces the clean 1024 mesh and the broken 1536 one, and
it is bit-identical to the single-graph path by test. So the difference is the
**input**, not the evaluation.

Voxel counts say the same thing independently. Pure surface scaling from 1024
to 1536 would be `(1536/1024)^2 = 2.25`; the measured ratio is
`9,715,171 / 3,440,673 = 2.82`. **25 % more voxels than a clean single-layer
surface needs** — and surplus voxels off the surface are exactly what creates
non-manifold junctions. The decoder is subdividing where it should not at 96³.

That is the plan's RoPE-extrapolation risk, no longer as an impression: a
96³ scaffold feeds coordinates up to 95 to a checkpoint that declares
`resolution: 64`, and the subdivision head is what degrades.

**Chunking encoder levels 1-3 was still worth doing.** The texture improved
visibly with it — the source image's cyan reaches the eyes now, which it did
not before — so that was a real defect. It was simply not the whole story, and
the earlier "refuted" verdict on it was wrong for the reason recorded above.

### Corrected once more: the tier is not broken, one run is

Three further runs on a slim subject (a punk figure) put the conclusion above
in a different light. Full set, all with every backend limit handled:

| subject | tier | seed | voxels | boundary | **non-manifold** | L4/L3 |
|---|---|---|---|---|---|---|
| punk | 1024 | 2026 | 1,526,670 | 0.505 % | 1.032 % | 4.041 |
| punk | 1536 | 2026 | 3,524,635 | 0.466 % | 0.699 % | 4.043 |
| punk | **1536** | 1517 | 3,343,769 | **0.135 %** | **0.357 %** | 4.014 |
| jester | 1024 | 42 | 3,440,673 | 0.171 % | 0.422 % | 4.001 |
| jester | **1536** | 42 | 9,715,171 | 0.921 % | **7.467 %** | **4.517** |

**The cleanest mesh in the set is a 1536 one**, and it beats every 1024 run.
Four of five runs sit between 0.36 % and 1.03 % with no tier dependence at all.
So "the tier fails on dense subjects" was wrong — what fails is the single
9.7M-voxel run, and the entry above should be read with that correction.

**L4/L3 is the early warning, and only that.** Every clean run sits at
4.00–4.04; the broken one is at 4.52. Above 4 the subdivision is thickening
rather than following a surface. It flags the catastrophe before mesh
extraction, but it does not rank the healthy runs — 4.041 gives 1.03 % while
4.043 gives 0.70 %.

**Where the boundary lies is unknown.** Every clean run is at or below 3.5M
voxels; the only failure is at 9.7M. Nothing was measured in between.

### The low-poly look is the seed, not the pipeline

A faceted-looking result at seed 2026 turned out to have **more** triangles than
the smooth one at seed 1517 (7,075,366 against 6,703,620), same tier, same code.
The SLAT simply came out angular. The metrics agree: the faceted run carries
twice the non-manifold rate and 3.5× the boundary edges, so "looks low-poly"
and "messier surface" are the same observation from two directions.

Worth keeping in mind before chasing a rendering artefact: with one vertex per
voxel the tessellation is always fine, so a faceted appearance is a statement
about the *shape*, never about triangle budget.

### The extractor is not the cause — now with a test

`tests/test_dual_grid` runs `fdg::extract` on analytic fields: a sphere at two
grid resolutions and a torus. All come out **closed, 2-manifold, with the right
Euler characteristic** (χ = 2 and 0), zero boundary and zero non-manifold edges,
and `drop_small_components` + `fill_holes` leave a closed mesh closed. It also
pins `fill_holes`' contract on a punctured surface: a generous limit closes the
loop, a limit below its length leaves it open.

So the 7.47 % non-manifold rate on the torn run is not the extractor
mishandling a fine grid — given a coherent voxel set it produces a clean
manifold at any resolution. The tear is in the **input**, which is the same
conclusion the CPU-decode and chunking experiments reached, now backed by a
test rather than by elimination.

This also closes a gap in `docs/VERIFICATION.md`, which credited "dual-grid
mesh extraction" to `test_marching_cubes` — a different extractor. The path
every generated mesh actually comes out of had no test.

### A warning for the one failure mode

The finest-level expansion ratio is now checked in `shape_dec_run` and warns
above 4.2 (measured: 4.00–4.04 on the four usable runs, 4.52 on the torn one).
It only warns — a generation this expensive should not be discarded on a
five-point heuristic, and the definitive numbers are the boundary and
non-manifold counts reported after extraction. But the one known way for a run
to be silently bad now says so while it is still running.

### Cost is occupancy, not resolution

The sparse-structure stage is a dense 16³ latent and costs the same for
everything (25–29 s in every run). Everything after it scales with how much of
the bounding cube the object's surface passes through, and that is fixed at the
32³ occupancy: 1,197 cells for the punk against 2,842 for the jester, a factor
of 2.37 that propagates unchanged through the HR scaffold (2.3×) and the HR flow
(39 s against 119 s).

Consequence worth knowing when planning runs: **the punk at 1536 (326 s) is
cheaper than the jester at 1024 (345 s).** The tier is not the cost driver.

## What is fixed, and what is not

**Fixed:** the aborts. A dense object completes at 1536 — 946 s, 9,715,171
vertices, 22,070,286 triangles.

**Not fixed, not diagnosed:** the result is visibly *worse* than the same object
at 1024, in colour and in surface holes. Three hypotheses were tested and all
three failed:

| hypothesis | test | outcome |
|---|---|---|
| column cap at encoder level 0 | two runs past it | correct materials — refuted |
| column cap at encoder level 1 | `TRELLIS2_SHAPE_ENC_CPU=1` | no change — **see the correction below** |
| 2 GB boundary in the resident tensor | `test_large_rows 10000000` | ops exact — refuted |

### The second row was not a clean refutation

Two single-component tests were treated as one two-component answer, and they
were not. `TRELLIS2_SHAPE_ENC_CPU=1` moved the encoder off the GPU while the
*decoder* still produced the geometry, and `TRELLIS2_SHAPE_DEC_CPU=1` moved the
decoder while encoder level 1 was still truncating:

| run | shape decoder | encoder level 1 | result |
|---|---|---|---|
| `SHAPE_ENC_CPU` | GPU | CPU, clean | bad |
| `SHAPE_DEC_CPU` | CPU | GPU, past the cap | slightly better — eyes visibly cleaner |

**Neither run had both clean**, so neither could isolate anything. The second
one is suggestive rather than conclusive: a decoder change alone moved the
geometry a little and left the materials largely as they were, which is what
partly-truncated texture latents would look like.

Rather than spend a third 40-minute run on it, the exposure was removed —
encoder levels 1-3 are now chunked too (above). The next 1536 run has no known
uncapped matmul anywhere and is the first that can actually answer the
question.

### The CPU decode did give one hard number

```text
[mesh] grid 1536: 9739619 verts, 22111834 tris,
       286917 boundary edges (0.9% of 31790138), 2341146 non-manifold edges
```

**2.3M non-manifold edges — 7.4 % of all edges carry more than two triangles.**
The holes at 0.9 % are the smaller problem by a wide margin: this mesh is
structurally broken, not merely open. Some non-manifold edges are inherent to
dual contouring over a sparse voxel set (voxels meeting only at an edge or
corner), so the number needs the 1024 baseline before it means anything — which
is one 6-minute run.

Also worth noting: CPU and GPU decodes do not agree exactly. 9,739,619 voxels
against 9,715,171, about 0.25 % apart. Expected — backend rounding shifts
marginal subdivision decisions — but it means "same seed" does not imply "same
mesh" across backends, and comparisons have to allow for it.

**And a correction to this entry's own earlier evidence.** The character run was
used twice to argue RoPE was not the cause. It should not have been: its 1536
mesh is 3,321,101 vertices, *smaller* than the jester's 1024 mesh at 3,440,673.
A sparse subject at 1536 lands in the size regime a dense one reaches at 1024,
so that run never entered the territory the tier exists for. Every run that did
reach it came out worse. One object is not a rate, but it is the only evidence
about that regime and it is negative.

## Open points

- The remaining test that separates model from code: `TRELLIS2_SHAPE_DEC_CPU=1`
  on the jester at 1536. The shape decoder produces the geometry *and* the
  encoder's input, so a clean CPU decode would mean a GPU defect at 9.7M voxels
  and a dirty one would mean the model. Cost: the decode moves from 66 s to an
  estimated 15-30 min, so ~30-45 min total — the flows stay on the GPU.
- Encoder level 1 is still unsplit and can pass the column cap (2,147,558 at
  1536). Refuted as the cause of the bad textures, but it is a real exposure and
  would need the per-stage ping-pong, since it carries four ConvNeXt blocks.
- If the CPU decode comes back dirty, the honest outcome is to leave the tier
  as it is — available, not in `T2_PIPE_AUTO`, with a note in `docs/PLAN.md`
  that it is currently worse than 1024 on dense objects.

## Proposed commit message

```text
Split the levels a dense cascade cannot launch in one graph

Two aborts on a dense 1536 object, both from a GPU dispatch being sized in
workitems in 32 bits rather than from the mul_mat column cap. Both offending
levels are now evaluated in voxel blocks - the decoder's per stage, so the
ConvNeXt chain needs no halo, and the encoder's level 0 across its two voxel
spaces. test_chunked_decode covers all three model paths at four block sizes,
bit-identical to the single-graph result.

The aborts are gone; the 1536 result is still worse than 1024 on a dense
object and that is not diagnosed. Detail, measurements and three refuted
hypotheses in docs/progress/1536-cascade_backend-limits.md.

No ABI change.
```
