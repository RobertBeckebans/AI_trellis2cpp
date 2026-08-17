# Progress — rocm-native-reference, phase 3: the SLAT dump, and what it exposed

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/rocm-native-reference.md`](../plan/rocm-native-reference.md) — phase 3
- **Branch:** `pixal3d`
- **Date:** 2026-08-16
- **Commits:** <to be added after review>

## Goal of the phase

Close the outstanding half of phase 3 — `dump_slat_reference.py` and
`dump_texture_reference.py` — as preparation for
[`docs/plan/pixal3d-proj-conditioning.md`](../plan/pixal3d-proj-conditioning.md),
whose phase 0 needs the same reference-dump harness, and for
[`backend-parity`](../plan/backend-parity.md) phase 1b, which depends on
the full cascade dump.

## What was done

**The SLAT reference dump exists.** `dumps/reference_slat.gguf`,
309,563,072 bytes, 21 tensors, seed 4321, `t=500`, resolution 512, run
natively on the R9700. It carries per-level decoder activations
(`lvl{0..3}.pre_up`, `lvl{0..3}.subdiv`), the active sets
(`lvl{0..4}.in_coords`), the flow forward, the sampled SLAT, and `out7` /
`out_coords` at 1,185,000 voxels.

Worth recording for D1: the reference selected `Conv backend: none` and
`Attention backend: sdpa`, i.e. the native-PyTorch sparse convolution and
`SDPBackend.MATH`, not the fast kernels. That is the *most* numerically
conservative configuration the reference offers, which strengthens it as
a yardstick — but it is still a ROCm-generated reference and D1 still
applies to every numeric row derived from it.

**`test_slat` went from `SKIP` to a real measurement**, and 14 of its 15
decoder taps are green (table under Verification). That is new coverage
for the whole shape-decoder chain: every level's active set is exact,
every level's features agree to ~1e-06, and every subdivision logit to
~7e-07.

**Two defects found on the way, one fixed, one still open.**

## Affected files

- `src/trellis2.cpp` — the decoder's column-limit guard now checks the
  backend before refusing (see Deviations); `splittable` no longer
  excludes tapped runs, and the chunked finest-level path captures the
  same taps the single-graph path does.
- `tests/test_slat.cpp` — a failing multi-channel output is now broken
  down per channel (rel-L2, cosine, mean, rms), so "wrong" can be told
  apart from "differently meant".
- `start_server.bat` — unrelated to this phase, carried along: the
  numerics knobs are documented per switch.
- `dumps/reference_slat.gguf`, `dumps/manifest_slat.json` — new, gitignored.

## Deviations / fixed along the way

**Fixed: the decoder refused to run on the CPU because of a GPU defect.**
The guard behind the unchunked fallback compared the level's voxel count
against `mul_mat_max_rows()`, which describes the ROCm/HIP column ceiling
— on the CPU backend there is none. `test_slat` pins the decoder to the
CPU by design, and with f32 weights the ceiling is 2^19 = 524,288, below
the 1,185,000 voxels a 512 decode reaches at its finest level. So the
decode aborted with a message naming a limit that did not apply, and the
decoder had never been validated at all. The encoder learned this exact
lesson in `e87069e` ("the warning now checks the backend"); this guard
did not get the same fix.

**Refuted: "the single-graph path is broken at large voxel counts."** The
first hypothesis for the remaining `out7` failure was that the finest
level diverges because taps exclude it from the chunked path. Removing
`!taps` from `splittable` so the chunked path runs instead produced
**bit-identical output** — same rel-L2, same worst index, same values. The
two paths are not merely equivalent in principle, they agree to the bit.
The evidence that had suggested otherwise (level 3 correct, level 4
wrong) was a misreading: `lvl3.pre_up` is captured *before* the output
layer and `out7` *after* it, so they were never comparable steps.

The change is kept because the exclusion's stated reason ("taps want the
full-length intermediates this path never forms") does not hold for the
finest level — it has no intermediate to tap — and because it is what
proved the two paths agree. It changes no numbers.

**Refuted: a layout, sign or channel-order mismatch in `out7`.** The
per-channel breakdown rules all three out; see Verification.

## Verification

`ctest -C Release -R "^slat$"`, ROCm runtime on `PATH` (without it the
test dies at `0xc0000135` in the loader, as `AGENTS.md` warns).

| Tap | rel-L2 | |
|---|---|---|
| `lvl{0..4}.in_coords`, `out_coords` | **0** | exact |
| `lvl{0..3}.pre_up` | 7.5e-07 … 1.1e-06 | OK |
| `lvl{0..3}.subdiv` | 4.8e-07 … 6.1e-07 | OK |
| `flow_t500_out` | 3.626e-06 | OK |
| `slat` (12-step sampler) | 8.774e-02 | within its documented 2e-01 gate |
| **`out7`** | **1.106** | **FAIL** |

Exact set sizes at every level mean no subdivision logit flipped sign, so
the geometry the decoder selects is right; only the values it writes into
it are wrong.

Per channel (`out7` is `[7, L]`, channel fastest, matching the
reference's `[L, 7]` row-major):

| ch | rel-L2 | cos | mean got | mean ref | rms got | rms ref |
|---|---|---|---|---|---|---|
| 0 | 0.953 | +0.456 | −0.007 | −0.283 | 0.60 | 0.75 |
| 1 | 0.993 | +0.206 | +0.002 | −1.027 | 0.60 | 1.61 |
| 2 | 0.985 | +0.316 | −0.007 | +0.513 | 0.56 | 0.97 |
| 3 | 0.874 | +0.648 | −10.22 | −7.08 | 19.23 | 17.87 |
| 4 | 1.128 | +0.201 | −10.63 | +7.68 | 19.45 | 25.61 |
| 5 | 1.228 | +0.124 | −12.03 | +7.34 | 19.52 | 23.03 |
| 6 | 1.011 | +0.176 | −0.066 | −7.79 | 5.26 | 12.97 |

Reading: every cosine is positive, so it is **not a sign flip**. The
channel groups behave differently from one another (ch0–2 small, ch3–5
large, ch6 its own), so it is **not a stride or layout mismatch** — that
would make all seven look alike. The group structure and the magnitudes
are right, which matches how `src/trellis2_capi.cpp` splits the seven
channels (offsets, intersection flags, one further channel). What is
wrong is the values, consistently: the reference's intersection channels
carry mixed signs (−7.1 / +7.7 / +7.3, "intersected frac 0.5654") where
the port's are uniformly −10 to −12, i.e. the port decides "no surface"
where the reference decides "surface".

## Open points

- **`out7` is unexplained.** The unverified span is now one short chain:
  level 3's `norm1 → silu → conv1` (producing `h1`), the host-side gather
  of surviving children, then level 4's `norm2 → silu → conv2 + skip`,
  the final `norm(1e-5)` and `output_layer`. Everything before it is
  green, including `lvl3.subdiv`, which hangs off the same `h` as
  `conv1` — so the up-block's *input* is proven right.
- **The production path is not implicated.** The server produces sound
  512 and 1024 meshes, and this is the same decoder. Whether the defect
  is in the port, in how the test feeds it, or in what the reference's
  `out7` means, is exactly what is not yet known.
- **The full cascade dump** (`dump_cascade_reference.py` *without*
  `--skip-hr-sampler`) is still outstanding. It is what `backend-parity`
  phase 1b depends on.
- **`dump_texture_reference.py`** is still outstanding. Its input pair is
  identified — `dumps/einstein_ref512.t2mesh` plus
  `generations/6daad1f22a18bbd0/input.img` (RGBA 1052²; the `--image`
  argument needs an alpha channel, which `dumps/einstein_pre.png` does
  not have) — but `tex_slat_flow_512` exists only as f16, and
  `test_texture` wants the flow alongside `shape_enc` and `tex_dec`.
- **The plan says the reference extra is `ref`; it is `rocm`.**
  `uv run --extra rocm` is the working invocation. Worth correcting in
  the plan's phase 0 text.

## Next phase

Tap `h1` — level 3's `conv1` output. It is already a named graph output
in the port (`outs.emplace_back( "h1", … )`), just not captured, and
adding the matching capture to `dump_slat_reference.py` bisects the
remaining chain exactly: if `h1` agrees, the defect is in level 4; if it
does not, it is in `conv1`. One dump re-run, one test run.

## Proposed commit message

```text
Produce the SLAT reference dump, and stop the decoder guard firing on the CPU

dumps/reference_slat.gguf takes tests/test_slat from SKIP to a measurement:
14 of 15 decoder taps green, active sets exact at every level, features and
subdivision logits at ~1e-06.

It could not run before. The guard behind the unchunked fallback compared the
level's voxel count against mul_mat_max_rows(), a ROCm ceiling the CPU backend
does not have - and the test pins the decoder to the CPU, where f32 weights put
that ceiling at 2^19, below the 1.185M voxels a 512 decode reaches. It now
checks the backend, the way the encoder's equivalent warning has since e87069e.

out7 still fails at rel-L2 1.106 and is not explained. A per-channel breakdown
in the test rules out a sign flip, a layout mismatch and a channel permutation:
the group structure and magnitudes are right, the values are not. Detail and
the two refuted hypotheses in
docs/progress/rocm-native-reference_3-slat-dump-and-out7.md.
```
