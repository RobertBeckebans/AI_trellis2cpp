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

## What is fixed, and what is not

**Fixed:** the aborts. A dense object completes at 1536 — 946 s, 9,715,171
vertices, 22,070,286 triangles.

**Not fixed, not diagnosed:** the result is visibly *worse* than the same object
at 1024, in colour and in surface holes. Three hypotheses were tested and all
three failed:

| hypothesis | test | outcome |
|---|---|---|
| column cap at encoder level 0 | two runs past it | correct materials — refuted |
| column cap at encoder level 1 | `TRELLIS2_SHAPE_ENC_CPU=1` | no change — refuted |
| 2 GB boundary in the resident tensor | `test_large_rows 10000000` | ops exact — refuted |

With every backend limit we know about handled, the gap remains. That points at
the model, which is the plan's original RoPE-extrapolation risk.

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
