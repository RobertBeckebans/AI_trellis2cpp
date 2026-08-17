# Progress — rocm-native-reference, phase 3: the SLAT dump, and the reference defect it exposed

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/rocm-native-reference.md`](../plan/rocm-native-reference.md) — phase 3
- **Branch:** `pixal3d`
- **Date:** 2026-08-16, resolved 2026-08-17
- **Commits:** `d799ead`, `46df6d9`, + this one

## Goal of the phase

Close the outstanding half of phase 3 — `dump_slat_reference.py` and
`dump_texture_reference.py` — as preparation for
[`docs/plan/pixal3d-proj-conditioning.md`](../plan/pixal3d-proj-conditioning.md),
whose phase 0 needs the same reference-dump harness, and for
[`backend-parity`](../plan/backend-parity.md) phase 1b, which depends on the
full cascade dump.

## Outcome in one line

`test_slat` **passes 18 of 18 decoder taps** against a CPU-generated reference.
The `out7` failure that looked like a port defect for most of two days is a
defect in the reference harness's own convolution, and only on ROCm.

## What was done

**The SLAT reference dump exists.** `dumps/reference_slat.gguf`, seed 4321,
`t=500`, resolution 512. It grew from 21 tensors to 26 as the bisection needed
more taps: per-level activations (`lvl{0..3}.pre_up`, `lvl{0..3}.subdiv`), the
active sets (`lvl{0..4}.in_coords`), the finest level's `pre_up`, the two
tensors the up-block hands across the level boundary (`in_hch` / `in_xch`), and
that block's `norm2` / `conv2`.

**`test_slat` went from `SKIP` to a real measurement.** It could not run at all
before: the decoder's column-limit guard refused on the CPU because of a ROCm
ceiling (fixed, see Deviations).

**A CPU-only reference settled it.** `--device cpu`, ~75 min wall clock against
~4 min on ROCm, 24.6 of 32 cores busy. Same script, same weights, same
implementation — only the device differs.

## Affected files

- `src/trellis2.cpp` — the decoder's column-limit guard checks the backend
  before refusing; the level-to-level hand-over is tapped; the share of the
  27-tap neighbourhood that contributes is logged per level.
- `scripts/dump_slat_reference.py` — captures the finest level's `pre_up` and,
  via forward hooks on the last up-block, `in_hch` / `in_xch` / `norm2` /
  `conv2`.
- `tests/test_slat.cpp` — a failing tap reports whole-tensor cosine and rms plus
  a per-channel breakdown; `TRELLIS2_DUMP_TAPS` writes one out for offline work.
- `dumps/reference_slat.gguf`, `dumps/reference_slat_cpu.gguf` — gitignored.

## Deviations / fixed along the way

**Fixed: the decoder refused to run on the CPU because of a GPU defect.** The
guard behind the unchunked fallback compared the level's voxel count against
`mul_mat_max_rows()`, the ROCm/HIP column ceiling — which the CPU backend does
not have. `test_slat` pins the decoder to the CPU by design, and with f32
weights that ceiling is 2^19 = 524,288, below the 1,185,000 voxels a 512 decode
reaches at its finest level. The encoder learned this in `e87069e`; this guard
had not.

**Eight hypotheses raised and refuted.** Recorded because each costs the same
day twice if it is not:

| Hypothesis | Refuted by |
|---|---|
| The single-graph path breaks at large voxel counts | Running the chunked path instead gives **bit-identical** output |
| Level 3 correct / level 4 wrong ⇒ level 4's arithmetic | Misread: `lvl3.pre_up` is *before* the output layer, `out7` *after* |
| The final `norm` + `output_layer` | `lvl4.pre_up` already fails, one step earlier |
| The skip connection | `skip` rms 3.21 against a 22.9 result; too small, and structurally identical to the reference |
| A missing or wrong conv bias | Per-channel ref/port std ratio spans 0.9–71.0; a bias is a constant offset |
| The neighbour map at the finest level | The port finds **20,262,654 of 31,995,000** slots — exactly what the reference's own coordinates give |
| The GGUF weights | rms identical to the checkpoint; layout `[Ci,27,Co]` confirmed against the converter and two independent ports |
| `conv_none.py`'s `weight.view(-1, C_out, C_in)` | Reproduces neither side — **and the file never ran** (see below) |

**The reference's convolution is not any upstream backend.**
`scripts/ref_common.py:214` defines its own sparse conv, registers it as
`conv_dispatch._backends["none"]`, and line 260 hard-assigns
`sp.config.CONV = "none"` — overriding the `SPARSE_CONV_BACKEND` env var that
line 35 sets only as a *default*. Consequences: the analysis of
`trellis2/modules/sparse/conv/conv_none.py` was on dead code, and an experiment
swapping in the Apple port's `conv_pytorch.py` silently measured nothing. The
tell was in the result — `lvl4.in_hch` came back identical **to the digit**
across the supposed backend change, and `conv1` runs through the same backend,
so it could not have been.

**Corrected framing:** at this stage the parity test compares two of *this
project's* implementations of the same operation and calls one of them the
reference.

## Verification

`test_slat` with the ROCm runtime on `PATH` (without it the test dies at
`0xc0000135` in the loader, as `AGENTS.md` warns).

**Against the CPU reference — everything green:**

| Tap | rel-L2 | |
|---|---|---|
| `lvl{0..4}.in_coords`, `out_coords` | **0** | exact |
| `lvl{0..3}.pre_up`, `lvl{0..3}.subdiv` | ≤ 1.1e-06 | OK |
| `lvl4.in_hch` / `lvl4.in_xch` | 3.681e-07 / 4.371e-07 | OK |
| `lvl4.pre_up` | **3.962e-07** | OK |
| `out7` | **4.318e-07** | OK |
| | | **RESULT: PASS**, 18/18 |

**The same taps against the ROCm reference:** `lvl4.pre_up` 0.979 FAIL, `out7`
1.103 FAIL.

**Where the two references part.** Both dumps carry `norm2` and `conv2` of the
last up-block, so the step can be checked in isolation against a direct
reimplementation of `conv2(silu(norm2(in_hch))) + repeat_interleave(in_xch)`
built from the reference's own tensors and the original checkpoint:

| | ROCm reference | CPU reference |
|---|---|---|
| `norm2` vs. affine-free LayerNorm | +1.000000 | +1.000000 |
| `pre_up` vs. `conv2 + skip` | +1.000000 | +1.000000 |
| **`conv2` vs. reimplementation** | **+0.164**, rms 22.6857 vs 4.6109 | **+1.000000**, rms 4.6109 vs 4.6109 |

Input verified identical, composition verified identical, output right on one
device and 4.9x too large on the other. The reimplementation also matches the
**port** at cos +1.000000, which is what clears it.

**Corroborating: the ROCm reference is not reproducible.** Across three runs of
one script at one seed, `intersected frac` moved 0.5654 → 0.5628 → 0.5527.

**And from outside the test entirely:** the server produces sound 512 and 1024
meshes from this same decoder. A 4.9x error at the finest level is not subtle.

## Open points

- **The defect in `scripts/ref_common.py`'s conv on ROCm is not yet pinned to a
  line.** The leading suspect is `out[hit] += contrib` — a masked scatter-add,
  which goes through atomics on the GPU. It explains both symptoms with one
  cause: over-accumulation reads as "too large", and a non-deterministic
  accumulation order reads as the run-to-run drift above. `torch.searchsorted`
  on ROCm is the second candidate. Neither is confirmed.
- **Numeric reference dumps should be produced on the CPU.** That is the
  practical rule this phase earns. The cost is ~75 min against ~4, and most of
  it is the sampler, which the decoder taps do not need — `test_slat` replaces
  the sampled SLAT with the dump's own (`slat = want`). Loading the SLAT from an
  existing dump and running only the decoder on the CPU would be both cheaper
  and a cleaner experiment, since only one variable moves.
- **`ref_common` should honour `SPARSE_CONV_BACKEND`** instead of overriding it,
  so an upstream backend can be compared against at all.
- **The full cascade dump** (`dump_cascade_reference.py` *without*
  `--skip-hr-sampler`) is still outstanding; `backend-parity` phase 1b depends
  on it. It should now be produced on the CPU.
- **`dump_texture_reference.py`** is still outstanding. Input pair identified —
  `dumps/einstein_ref512.t2mesh` plus `generations/6daad1f22a18bbd0/input.img`
  (RGBA 1052²; `--image` needs an alpha channel, which `dumps/einstein_pre.png`
  lacks) — but `tex_slat_flow_512` exists only as f16.
- **The dumps are 1.6 GB each.** `norm2`, `conv2`, `in_hch` and `in_xch` are
  investigation aids; only the finest level's `pre_up` earns a permanent place.
- **The plan says the reference extra is `ref`; it is `rocm`.**
  `uv run --extra rocm` is the working invocation.

## What this means for the plans

`rocm-native-reference` D1 warned that a ROCm-generated reference is not
independent of the port, because a *shared* defect would cancel and the test
would go green while both sides were wrong. What happened is the mirror image
and worse to diagnose: a defect only the reference has, which presents as a port
defect and sends the investigation into the port for two days.

`backend-parity` D2 ("PyTorch is the reference") now has a concrete procedure
attached: on this hardware, produce it on the CPU or it is not a reference.

## Next phase

Regenerate the cascade and texture dumps on the CPU, then resume
`backend-parity` phase 1b. Pinning the ROCm defect in `ref_common`'s conv is
worth a short separate look — a minimal reproducer of the masked scatter-add
would be reportable upstream, and would tell whether anything else in the port
touches the same op.

## Proposed commit message

```text
Clear the port of out7: the reference's convolution is wrong on ROCm, not ours

A CPU-generated reference takes tests/test_slat to 18 of 18 decoder taps, with
lvl4.pre_up at 3.962e-07 and out7 at 4.318e-07 - the two that failed at 0.979
and 1.103 against the ROCm one.

Both dumps carry norm2 and conv2 of the last up-block, so the step isolates:
input identical on both devices (cos +1.000000), composition identical
(pre_up = conv2 + skip, +1.000000), and the convolution right on the CPU
(+1.000000) but 4.9x too large and uncorrelated on ROCm (+0.164). The same
reimplementation matches the port bit for bit, which is what clears it.

The convolution in question is not upstream's: scripts/ref_common.py defines its
own, registers it as backend "none" and hard-assigns config.CONV, overriding the
SPARSE_CONV_BACKEND env var. So conv_none.py never ran and an experiment
swapping backends measured nothing. Leading suspect is the masked scatter-add
out[hit] += contrib, which also explains why the ROCm reference is not
reproducible run to run (intersected frac 0.5654 -> 0.5628 -> 0.5527).

Practical rule this earns: numeric reference dumps go on the CPU.

Detail and the eight refuted hypotheses in
docs/progress/rocm-native-reference_3-slat-dump-and-out7.md.
```
