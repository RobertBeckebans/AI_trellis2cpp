# Progress — backend-parity: the 1024 tier, and why exact attention could not reach it

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/backend-parity.md`](../plan/backend-parity.md) — phase 0b
- **Branch:** `pixal3d`
- **Date:** 2026-08-17/18
- **Commits:** this one

## Goal of the phase

Phase 0b established *why* the 1024 cascade diverges on ROCm: the HR flow is
forced onto `ggml_flash_attn_ext`, whose HIP kernels accumulate in F16 and ignore
`ggml_flash_attn_ext_set_prec()`. It did not establish what to do about it. The
reviewer's own report was the trigger: 512 produces clean meshes, 1024 produces
mostly broken ones, and the Apple port does better at that tier.

## Outcome in one line

Blocking the exact attention path over the **query** axis puts it within reach at
HR token counts, and the finest-level expansion ratio at 1024 moves from
**3.43/3.55 to 4.06** — into the healthy band for the first time.

## The chain, top to bottom

1. `sdpa_auto` gates exact attention on whether the `[Lk, Lq, H]` score matrix
   fits a cap (default 1 GiB). At 1024 the HR flow carries **23,358 tokens** and
   12 heads, so self-attention wants **26.2 GB** in one allocation. It never
   passes the gate.
2. So the HR flow runs on flash, in F16, at 89.8 % latent sign agreement against
   99.9 % for exact (measured in phase 0b).
3. The decoder thresholds subdivision logits at zero. A noisy latent flips the
   *marginal* voxels — which are exactly the ones that make up thin structures.
4. Result: massive parts survive, thin parts die. On a TV-on-a-stand fixture the
   television came out intact while the arched wooden stand became a perforated
   plate, then four floating stubs. The finest-level expansion ratio recorded
   3.55 and 3.43 against a healthy band of 4.00–4.04.

## The fix

**Split the score matrix along the query axis.** This is exact and needs no
correction term: `soft_max` runs along the key axis, so every query row is
independent of every other. Only a split along *keys* would need flash's online
rescaling, which is the whole of flash's complexity and none of it is required
here.

Peak score memory becomes `Lk*B*H*4` instead of `Lk*Lq*H*4`. At the 1 GiB default
that is 957 rows per block and 25 blocks for the 1024 cascade.

**Blocking, not a bigger cap, is what mattered.** Raising the cap far enough to
let the 26 GB matrix through does allocate — `ggml_soft_max_ext` works in place,
so the earlier estimate of 52 GB was too pessimistic — but 26 GB beside the model
does not fit in 32 GB of VRAM, so it paged to host memory and cost **585 s per
step against flash's 28.9 s**. In 1 GiB blocks the same arithmetic runs at
**86.7 s per step**, 6.7x faster than unblocked exact.

## Results

One fixture, one seed (42), 1024 tier, R9700 / ROCm.

| | flash (default) | exact, `EXACT_MAX_MB=2048` | **exact, blocked** |
|---|---|---|---|
| **finest-level expansion** | 3.547 / 3.553 | 3.432 / 3.436 | **4.055 / 4.058** |
| voxels, finest level | 4,311,749 | 3,816,164 | **7,923,512** |
| triangles | 5,816,108 | 4,334,452 | **15,873,830** |
| boundary edges | 0.7087 % | 0.9065 % | **0.6555 %** |
| non-manifold edges | 0.873 % | 1.003 % | **0.690 %** |
| 1024 SLAT sampling | 310.1 s | 310.1 s | 860.7 s |
| total | 716.8 s | 766.1 s | 1741.1 s |

The edge counters *fell* while the triangle count nearly tripled. That is the
result that rules out "denser noise": more geometry at better topology means the
earlier runs were losing surface, not that this one is inventing it.

Cost is **2.3x** end to end. Most of it is the two sampling stages (1024 SLAT
310 → 861 s, material 254 → 647 s); mesh extraction rises 8.5 → 30.3 s purely
because there is more of it.

## Affected files

- `src/trellis2.cpp` — `sdpa_auto` builds the score matrix in query blocks;
  `TRELLIS2_SDPA_BLOCK_MB` (default 1 GiB) sizes them; `trellis2_effective_config`
  reports `sdpa_block_mb`, which is also how a manifest shows whether a run used
  this code.

## Deviations / fixed along the way

**Fixed: the blocking aborted at small block sizes.** `TRELLIS2_SDPA_BLOCK_MB=4`
died in `ggml_new_object: not enough space in the context's memory pool` →
`GGML_ASSERT(obj_new)`. Each block costs ~5 graph nodes and the flow graphs are
built with a fixed budget of 32,768, which the blocking knows nothing about. A
64-block ceiling now grows the block instead of failing; the 1024 cascade asks
for 25, so it only ever fires on a pathological setting.

**Six hypotheses raised and refuted.** Recorded because each of them looked
convincing at the time:

| Hypothesis | Refuted by |
|---|---|
| The alpha matte is wrong — it fills the arch openings and drops the VCR | True, but not the cause: 512 uses the *same* matte and builds both correctly. The model reads RGB, not the mask |
| Exact attention at 1024 cannot allocate (~52 GB) | It allocates. `soft_max_ext` is in place, so it is 26 GB — and it pages rather than fails |
| Chunking / the `mul_mat` column limit is implicated | Every over-cap level in the logs is marked `(chunked)`; the one unhandled warning needs `Lc > 2^21` and `Lc` is 1.2M |
| `soft_max` exceeds a 65,535 grid dimension on HIP | `ggml/src/ggml-cuda/softmax.cu:340` launches `block_nums(ne01, ne02, ne03)` = (23358, 12, 1). Both far under |
| Making cross-attention exact would help | Measured: +7 % time, result still broken. Only self-attention matters, and it stayed on flash |
| Exact attention *destroyed* the mesh (the `EXACT_MAX_MB=2048` run) | Both runs are broken — 3.55 and 3.43, both outside the healthy band. The difference between two wrecks is a chaotic sampler responding to a perturbation |

**The measurement that unlocked it.** Making the *512* self-attention exact
(1.26 GB, no memory pressure) moved that stage from 42.63 s to 76.82 s —
**1.8x, not 20x**. That is what said the exact kernel is healthy and the 20x at
1024 was paging, which is what made blocking worth building rather than
abandoning the exact path.

## Verification

`tests/test_ss_flow_forward` runs the exact path by default, so forcing a
multi-block split against an unchanged fp32 reference tests the blocking
directly:

| `TRELLIS2_SDPA_BLOCK_MB` | blocks | rel-L2 vs. fp32 reference | |
|---|---|---|---|
| 1024 (default) | 1 | 3.050e-06 | PASS |
| 16 | many | 3.042e-06 | PASS |
| 4 | many | 3.098e-06 | PASS |
| 1 | many (capped at 64) | 3.083e-06 | PASS |
| — (`TRELLIS2_SDPA_FLASH=1`) | — | **5.198e-02** | **FAIL** |

Blocked exact is the same as unblocked exact; what moves between block sizes is
reduction order, three orders of magnitude below the 2e-03 gate. The flash row is
the documented contrast and is why this matters at all.

`ctest --test-dir build -C Release -LE model`: **10/10, no skips.**

`dino` fails, and does not fail because of this. Its first divergence is `embd` —
`tap("embd", h)` at `src/trellis2.cpp:1686`, the first `sdpa` call at `:1705`, so
no attention is involved. It is a known open item
(`docs/plan/rocm-native-reference.md:306`, "dino's patch embedding is ~f16
accurate").

## Open points

- **One fixture, one seed.** The mechanism is confirmed by the expansion ratio,
  but "1024 is fixed" needs more than one sample. Two or three more objects
  before any default moves.
- **The default is unchanged on purpose.** The 1 GiB gate measured *memory*, and
  blocking has removed that wall; what is left is a time question, and there is
  exactly one measurement of it. `TRELLIS2_SDPA_EXACT=1` selects the new path
  today.
- **An env var is the wrong interface for this.** It is a per-job quality
  decision and needs a server restart to change. It belongs in the UI, which
  means bundling `t2_generate`'s parameters into a struct first — the function
  already carries 15 arguments, exactly purego's limit.
- **The subdivision warning's lower threshold is too permissive.** It fires below
  3.5 and let 3.547 through. Three points now bracket it: 3.432 broken, 3.547
  broken, 4.055 healthy. 3.8 separates them.
- **Why the 512 tier was fine all along** is worth one sentence somewhere: its
  self-attention score is 1.26 GB, so it misses the 1 GiB gate by 18 % and also
  ran on flash — but at 5,121 tokens flash's F16 accumulation does not move
  enough voxels to matter.
- **The alpha matte defect is real and unfixed.** Automatic background removal
  fills the arch openings and removes the VCR on a white-on-white image with
  through-holes. It cost this investigation an hour and will cost someone else
  more on a harder object.

## What this means for the plans

`backend-parity` phase 0b diagnosed the cause and stopped there. The cause is now
confirmed at the mesh level and has a fix. The plan's D3 question — how tight a
backend-comparison gate has to be — is unaffected; what changes is that ROCm at
1024 is no longer disqualified by construction.

## Next phase

Bundle `t2_generate`'s parameters into a `t2_gen_options` struct (ABI 18 → 19),
add the attention mode to it, and expose it in the UI. Then re-measure on two or
three further fixtures and decide the default from data.

## Proposed commit message

```text
Build exact attention in query blocks, and reach the 1024 tier with it

The HR cascade could not use exact attention: at 23,358 tokens and 12 heads its
score matrix is 26 GB in one allocation, so sdpa_auto's 1 GiB gate always chose
flash - whose HIP kernels accumulate in F16 and ignore GGML_PREC_F32. That is 89.8%
latent sign agreement against 99.9%, and the decoder thresholds subdivision at
zero, so the voxels it flips are the marginal ones that thin structures are made
of. Massive parts survived, thin parts died.

Splitting the score matrix along the QUERY axis is exact with no correction term,
because soft_max runs along keys and every query row is independent. Only a split
along keys would need flash's online rescaling. Peak memory becomes Lk*B*H*4, so
TRELLIS2_SDPA_BLOCK_MB (default 1 GiB) bounds it at 25 blocks of 957 rows instead
of one 26 GB piece.

The blocking is the part that mattered. A cap raised far enough does allocate -
soft_max_ext works in place - but 26 GB beside the model pages to host memory and
costs 585 s per step against flash's 28.9. Blocked: 86.7 s.

Measured on one fixture at 1024, against flash: finest-level expansion 3.55 -> 4.06
(healthy band is 4.00-4.04), 4.3M -> 7.9M voxels, 5.8M -> 15.9M triangles, and the
edge counters FELL - non-manifold 0.87% -> 0.69%, boundary 0.71% -> 0.66%. More
geometry at better topology, so the earlier runs were losing surface. Cost is 2.3x
end to end.

test_ss_flow_forward already runs the exact path, so a forced multi-block split
tests it directly: 3.050e-06 in one block, 3.042/3.098/3.083e-06 in many, against
5.198e-02 for flash. ctest -LE model 10/10.

The default gate is deliberately unchanged: it measured memory, blocking removed
that wall, and what is left is a time question with one measurement behind it.
TRELLIS2_SDPA_EXACT=1 selects the new path until the UI can.

Detail, the six refuted hypotheses and the numbers in
docs/progress/backend-parity_1024-exact-attention.md.
```
