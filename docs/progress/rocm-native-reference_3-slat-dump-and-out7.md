# Progress — rocm-native-reference, phase 3: the SLAT dump, and what it exposed

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/rocm-native-reference.md`](../plan/rocm-native-reference.md) — phase 3
- **Branch:** `pixal3d`
- **Date:** 2026-08-16, extended 2026-08-17 with the bisection
- **Commits:** <to be added after review>

## Goal of the phase

Close the outstanding half of phase 3 — `dump_slat_reference.py` and
`dump_texture_reference.py` — as preparation for
[`docs/plan/pixal3d-proj-conditioning.md`](../plan/pixal3d-proj-conditioning.md),
whose phase 0 needs the same reference-dump harness, and for
[`backend-parity`](../plan/backend-parity.md) phase 1b, which depends on
the full cascade dump.

## What was done

**The SLAT reference dump exists.** `dumps/reference_slat.gguf`, seed 4321,
`t=500`, resolution 512, run natively on the R9700. It started at
309,563,072 bytes / 21 tensors and grew to 1,598,843,264 / 26 as the
bisection needed more taps.

Worth recording for D1: the reference selected `Conv backend: none` and
`Attention backend: sdpa`, i.e. the ROCm fork's native-PyTorch sparse
convolution and `SDPBackend.MATH`, not the fast kernels. That reads as the
most conservative configuration available — and by the end of this entry it
is the prime suspect instead.

**`test_slat` went from `SKIP` to a real measurement**, and 16 of its 18
decoder taps are green. That is new coverage for the whole shape-decoder
chain: every level's active set is exact, every level's features agree to
~1e-06, and every subdivision logit to ~7e-07.

**`out7` failed, and the bisection that followed cleared the port.** Five
rounds of taps narrowed a half-pipeline span to a single operation, and then
showed that operation is right in the port and unreproducible in the
reference. Detail under Verification.

**Two defects found on the way**, one fixed (the CPU guard), one still open
and no longer believed to be ours.

## Affected files

- `src/trellis2.cpp` — the decoder's column-limit guard now checks the
  backend before refusing; the level-to-level hand-over (`in_hch` / `in_xch`)
  is tapped; the share of the 27-tap neighbourhood that contributes is
  logged per level under `TRELLIS2_TIMING`.
- `scripts/dump_slat_reference.py` — captures the finest level's `pre_up`
  (the loop only did so for levels with an up-block) and, via forward hooks
  on the last up-block, `in_hch` / `in_xch` / `norm2` / `conv2`.
- `tests/test_slat.cpp` — a failing tap now reports whole-tensor cosine and
  rms, plus a per-channel breakdown for `out7` / `pbr`; `TRELLIS2_DUMP_TAPS`
  writes a failing tap out for offline analysis.
- `dumps/reference_slat.gguf`, `dumps/manifest_slat.json` — new, gitignored.

## Deviations / fixed along the way

**Fixed: the decoder refused to run on the CPU because of a GPU defect.**
The guard behind the unchunked fallback compared the level's voxel count
against `mul_mat_max_rows()`, which describes the ROCm/HIP column ceiling —
on the CPU backend there is none. `test_slat` pins the decoder to the CPU by
design, and with f32 weights the ceiling is 2^19 = 524,288, below the
1,185,000 voxels a 512 decode reaches at its finest level. So the decode
aborted naming a limit that did not apply, and the decoder had never been
validated at all. The encoder learned this exact lesson in `e87069e` ("the
warning now checks the backend"); this guard did not get the same fix.

**Six hypotheses raised and refuted.** Recorded because each one costs the
same day twice if it is not:

| Hypothesis | Refuted by |
|---|---|
| The single-graph path breaks at large voxel counts (taps exclude the chunked one) | Running the chunked path instead gives **bit-identical** output |
| Level 3 correct / level 4 wrong ⇒ level 4's arithmetic | Misread: `lvl3.pre_up` is *before* the output layer, `out7` *after* — never comparable steps |
| The skip connection is missing or wrong | `skip` rms is 3.21 against a 22.9 result; too small to account for it, and the port's post-skip mean is ~0 as expected |
| A missing or wrong conv bias | The per-channel ref/port std ratio spans 0.9–71.0; a bias is a constant offset |
| The neighbour map is wrong at the finest level | The port finds **20,262,654 of 31,995,000** slots, matching a count computed independently from the reference's own coordinates, exactly |
| The GGUF weights are wrong or transposed | rms identical to the checkpoint; layout `[Ci,27,Co]` confirmed against the converter and two independent ports |

**Reverted from the previous commit:** dropping `!taps` from `splittable`.
The exclusion was right — the chunked path never forms `pre_up`, which is
the one tensor that bisects this span. It changes no numbers either way.

## Verification

`ctest -C Release -R "^slat$"`, ROCm runtime on `PATH` (without it the test
dies at `0xc0000135` in the loader, as `AGENTS.md` warns).

| Tap | rel-L2 | |
|---|---|---|
| `lvl{0..4}.in_coords`, `out_coords` | **0** | exact |
| `lvl{0..3}.pre_up` | 7.5e-07 … 1.1e-06 | OK |
| `lvl{0..3}.subdiv` | 4.8e-07 … 6.1e-07 | OK |
| `lvl4.in_hch` | **6.141e-07** | OK |
| `lvl4.in_xch` | **7.556e-07** | OK |
| `flow_t500_out` | 3.626e-06 | OK |
| `slat` (12-step sampler) | 8.774e-02 | within its documented 2e-01 gate |
| `lvl4.pre_up` | 0.979 | FAIL |
| `out7` | 1.103 | FAIL |

Exact set sizes at every level mean no subdivision logit flipped sign: the
geometry the decoder selects is right, only the values written into it differ.

**Where the break is.** The finest level computes
`conv2(silu(norm2(in_hch))) + repeat_interleave(in_xch)`
(`SparseResBlockC2S3d._forward`). Taking it apart against the reference's own
tensors:

| Stage | reference rms | independent reimplementation | cos |
|---|---|---|---|
| `norm2` | 1.0000 | 1.0000 | **+1.000000** |
| `pre_up` vs. `conv2 + skip` | 22.9039 | 22.9039 | **+1.000000** |
| `conv2` | 22.6857 | 4.6109 | **+0.164** |

So the reference's own `norm2` is a plain affine-free LayerNorm, its own
`pre_up` really is `conv2 + skip`, and the disagreement is confined to the
convolution.

**And the convolution is right in the port.** Reimplementing the documented
formula in Python — reference tensors in, original checkpoint weights,
27-tap submanifold gather — reproduces the **port**, not the reference:

```
cos(reimplementation, port_conv2) = +1.000000   rms 4.6109 = 4.6109
cos(reimplementation, ref_conv2)  = +0.164170
```

The reference's `conv2` could not be reproduced from any plausible input
(`silu(norm2)` +0.164, `silu(in_hch)` +0.161, `in_hch` +0.179, `norm2`
+0.176 — a plateau, so the input is not the variable), nor from `conv_none`'s
own `weight.view(-1, C_out, C_in)` (+0.060).

**The strongest independent argument is outside the test.** The server
produces sound 512 and 1024 meshes from this same decoder. An error of ~5x at
the finest level is not subtle; it would destroy every mesh.

**The reference is not bit-reproducible.** Across two runs of the same script
with the same seed, `intersected frac` moved 0.5654 → 0.5628 and `offsets
mean` 0.4550 → 0.4525. The port's figures are stable across runs; the
reference's are not.

## Open points

- **`out7` is now the reference's problem, not the port's**, and the
  mechanism is still unnamed. Everything around the convolution checks out at
  cos +1.000000; the convolution itself, as the dump ran it, does not match
  any reading of its own source. The suspect is `conv_none`, the ROCm fork's
  addition, whose header calls it "MUCH SLOWER … Use for debugging" and which
  ships a second, unreachable implementation beside it.
- **This is D1 arriving in practice.** The plan warned that a
  ROCm-generated reference is not independent of the port for numeric taps.
  It is worse than the anticipated case: not a shared defect cancelling out,
  but a reference-only one that reads as a port defect.
- **The full cascade dump** (`dump_cascade_reference.py` *without*
  `--skip-hr-sampler`) is still outstanding, and is what `backend-parity`
  phase 1b depends on.
- **`dump_texture_reference.py`** is still outstanding. Its input pair is
  identified — `dumps/einstein_ref512.t2mesh` plus
  `generations/6daad1f22a18bbd0/input.img` (RGBA 1052²; `--image` needs an
  alpha channel, which `dumps/einstein_pre.png` lacks) — but
  `tex_slat_flow_512` exists only as f16.
- **The dump has grown to 1.6 GB.** `norm2`, `conv2`, `in_hch` and `in_xch`
  are investigation aids; only the finest level's `pre_up` earns a permanent
  place once this is settled.
- **The plan says the reference extra is `ref`; it is `rocm`.**
  `uv run --extra rocm` is the working invocation.

## Next phase

Regenerate the dump with a different sparse-conv backend and see whether
`out7` falls into line. `docs/ref/trellis2-apple/` (MIT, same Microsoft
origin) ships `conv_pytorch.py`: an independent CUDA-free implementation that
permutes `(Co,Kd,Kh,Kw,Ci) → (K,Ci,Co)` explicitly where `conv_none` takes a
raw `view`. It is a configuration change, not code.

If the port then agrees with the reference, `conv_none` is the culprit and
the port was never wrong — and `backend-parity` gains a reference whose
geometry path contains no vendor kernel at all, which is what its D2 asks
for.

## Proposed commit message

```text
Bisect the decoder's finest level, and clear the port of the out7 failure

New taps split the one span that was opaque. The reference dump gains the
finest level's pre_up (the loop only captured it for levels with an up-block),
the two tensors the up-block hands across the level boundary, and norm2/conv2
of that block. The port taps the matching pair after its host-side gather;
tests/test_slat reports whole-tensor cos/rms and a per-channel breakdown for a
failing tap, and TRELLIS2_DUMP_TAPS writes one out for offline work.

With those, out7 is no longer the port's. lvl4.in_hch and lvl4.in_xch match at
6.1e-07 and 7.6e-07, the neighbour map matches the reference coordinates
exactly (20,262,654 of 31,995,000 slots), and reimplementing the documented
formula from the reference's own tensors and the original checkpoint reproduces
the port bit for bit - cos +1.000000. The reference's conv2 output cannot be
reproduced from any of them, while its norm2 and its pre_up = conv2 + skip both
check out at cos +1.000000. The break is inside sp.SparseConv3d as the dump ran
it, on the ROCm fork's conv_none backend, which its own header calls debug-grade.

Reverts last commit's splittable change: the taps exclusion it dropped was
right after all, because the chunked path never forms pre_up.

Also logs how much of the 27-tap neighbourhood contributes per level - the
cheapest tell that a neighbour map is wrong.

Detail, the six refuted hypotheses and the next experiment in
docs/progress/rocm-native-reference_3-slat-dump-and-out7.md.
```
