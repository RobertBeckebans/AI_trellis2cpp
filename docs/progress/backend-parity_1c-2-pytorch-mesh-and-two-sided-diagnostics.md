# Progress — backend-parity, phases 1c and 2: a PyTorch mesh in the viewer, and diagnostics that fire both ways

- **Issue:** none — tracked by plan key
- **Plan:** [`docs/plan/backend-parity.md`](../plan/backend-parity.md) — Phases 1c and 2
- **Branch:** `pytorch-comparison`
- **Date:** 2026-08-16
- **Commits:** <to be added after review>

## Goal of these phases

Phase 1c: put a mesh produced by the reference implementation under
`generations/`, next to the ROCm, Vulkan and CPU runs of the same image, so the
question "which of these is right" can be looked at instead of inferred from
counters. Phase 2: make the decoder's health check report collapse as well as
runaway, because the worst failure on record passed the one-sided version
silently.

## Phase 1c — the reference generation

### The reference package now lives in the repo root

`docs/ref/TRELLIS-2_rocm/trellis2/` copied to `trellis2/` and **gitignored**.
It is MIT, so there is no license conflict, but it is reference material we run
against rather than code we adopt, and `CLAUDE.md` is explicit that nothing
from the TRELLIS reference enters our sources. Keeping it untracked is what
matches that: importable, never committed.

`scripts/ref_common.py` now resolves `TRELLIS2_PY` to the repo root when a
`trellis2/` package is there, falling back to the container's `/trellis2`.
Both environments run the same scripts unchanged.

### It runs natively on Windows/ROCm

Only `matplotlib` had to be added to `pyproject.toml` — the upstream pipeline
module imports it at module level for its sparse-structure visualisations, so
`Trellis2ImageTo3DPipeline` cannot be imported without it even when nothing is
plotted. Everything else was already in the Dockerfile pip set.

The compiled CUDA extensions (`o_voxel`, `flex_gemm`, `cumesh`, `flash_attn`)
are stubbed by `ref_common` as before, and the sparse conv/attention run
through the pure-torch backends. Verified: the real
`Trellis2ImageTo3DPipeline` and `trellis2.representations` import and run under
`uv run --extra rocm`.

### `scripts/ref_generate.py`

Mirrors `Trellis2ImageTo3DPipeline.run()` for pipeline types `512` and
`1024_cascade`, but stops one step earlier — `run()` hands the decoder's
7-channel output to o_voxel's CUDA mesher, which does not exist here. Those 7
channels are dumped instead.

Three faithfulness points worth stating, because each of them is a place the
artefact could have silently stopped being a reference:

- **RNG order matches `run()`**: one `torch.manual_seed(seed)`, then the
  sparse-structure noise, then the shape-SLAT noise, all drawn on CPU. So the
  seed selects the noise the reference implementation would have used.
- **True fp32** via `ref_common.setup()` (TF32 off, reduced-precision SDPA
  kernels off), as every other reference dump in this repo.
- **The conditioning path is the validated one** — the same manual DINOv3
  encode `scripts/dump_dino_reference.py` uses, whose output the C++ DINO stage
  is already gated against. Not a second implementation of it.

Preprocessing is the alpha branch of `preprocess_image`; an opaque image is
rejected rather than silently taking a different path, because the driver does
not load the rembg model.

### `examples/dual_grid_cli.cpp`

The reference cannot mesh its own output here, so its voxels go through **our**
extractor — `examples/flexible_dual_grid.h`, the same code every backend uses.
That is the point rather than a compromise: with one mesher on both sides, what
is left between this artefact and a trellis2.cpp generation is the network.

The post-extraction steps mirror `trellis2_capi.cpp`'s generate path exactly
(`drop_small_components`, grid-scaled `fill_holes`, `vertex_normals`,
`orient_faces`). A reference mesh that skipped them would differ from every
generation for a reason unrelated to the backends. The topology counters print
unconditionally — this tool exists to produce a comparison and the comparison
is those numbers.

Input format `FDGVOX01`: `magic[8] | u32 n | u32 grid | f32 margin |
i32[3n] coords | f32[7n] feats`. Minimal on purpose; it carries one decoder
output to one extractor.

### `scripts/ref_publish_generation.py`

Writes the `generations/<id>/` layout `server/persistence.go` restores at
startup: `mesh.t2mesh`, `input.img`, `source.img`, `manifest.json` (v1), with
the same build-a-temp-directory-then-rename discipline as `persistJob`, so the
server never sees a half-written generation.

The device string is `PyTorch reference, our extraction [PyTorch]`. The
bracketed tag is what `deviceTag()` in the viewer extracts for the tile badge,
so the history strip reads `512 · PyTorch` and the tooltip carries the full
sentence. A mesh labelled like a backend run would be read as one.

### Result

Einstein, seed 42, pipeline type `512`, on the R9700 (decoder on CPU, as in the
C++ port):

```
scaffold:            2673 voxels at 32^3 (8.16 % occupancy)
decode level 0..4:   11211 -> 47519 -> 205316 -> 920630 -> 920630
out7:                920630 voxels, intersected fraction 0.3419
mesh:                920,630 verts, 1,896,504 tris
                     27,755 boundary edges (0.9809 %), 48,043 non-manifold
wall clock:          436 s  (SS 137 s, SLAT 82 s, decode 210 s)
```

Published as `generations/6daad1f22a18bbd0`.

### What it already says, and what it does not

Against our own runs from the **same input file and seed** at 512:

| | verts | tris | tris/vert |
|---|---|---|---|
| PyTorch reference | 920,630 | 1,896,504 | 2.06 |
| trellis2.cpp CPU | 711,670 | 1,428,652 | 2.01 |
| trellis2.cpp ROCm | 711,680 | 1,428,668 | 2.01 |
| trellis2.cpp Vulkan | 728,910 | 1,464,386 | 2.01 |

The reference carries **29.4 % more vertices** than our CPU run.

**A correction to an earlier draft of this document.** It compared against
runs measuring 724,634 vertices and reported 27.0 %. Those runs were made with
`background = AUTO`; the reference driver takes `preprocess_image`'s alpha
branch, which is `background = KEEP`. Different preprocessing means a different
conditioning image, so that comparison mixed two setups. The table above is the
matched one — every row `background = KEEP`, seed 42, same input file. The 1024
comparison below was unaffected: all of those runs were already KEEP.

Boundary and non-manifold counts are not in the persisted manifests, so the
only 512 topology numbers here are the reference's own (0.98 % boundary,
48,043 non-manifold).

### Then at 1024

`--pipeline-type 1024_cascade`, same image and seed, 1,716 s
(`generations/73f701b208b57c39`). Scaffold 2,673 → 11,203 HR tokens → 3,674,840
voxels. The topology counters here *are* comparable, because the backend runs
were measured with the same extractor and the same counters:

| | verts | tris | boundary | non-manifold |
|---|---|---|---|---|
| **PyTorch reference** | **3,674,840** | 7,382,010 | **0.2728 %** | **72,713** |
| trellis2.cpp CPU | 2,854,472 | 5,732,416 | 0.1157 % | 43,204 |
| trellis2.cpp ROCm | 2,838,357 | 5,689,864 | 0.0415 % | 22,814 |
| trellis2.cpp Vulkan | 2,940,217 | 5,907,996 | 0.1716 % | 53,593 |

**The vertex gap holds**: +28.8 % over our CPU run, after +29.4 % at 512. But
that consistency is weaker evidence than it looks — both reference runs descend
from the *same* 32³ scaffold and the same LR SLAT, so they are not independent
samples. What it rules out is that the 512 figure was a decoder fluke; it does
not by itself establish a systematic difference.

### The finding that matters most: ROCm reproduces the CPU at 512 and stops at 1024

Putting the two matched tiers side by side, as deviation from the CPU run:

| | 512 | 1024 |
|---|---|---|
| ROCm | **+0.0014 %** (10 vertices of 711,670) | **−0.56 %** (16,115 vertices) |
| Vulkan | +2.42 % | +3.00 % |
| PyTorch reference | +29.4 % | +28.8 % |

**ROCm and the CPU are ten vertices apart at 512** — 0.0014 %, and the reviewer
reports the two meshes as visually indistinguishable, which 10 vertices in
711,670 fully accounts for. At 1024 the same pair is 400× further apart.
Vulkan, by contrast, is off at both tiers and barely more so at the higher one.

That splits the problem in a way none of the earlier measurements could:

- **ROCm's divergence is not a general numeric-accuracy problem.** If it were,
  it would show at 512 too. It does not, essentially at all.
- **Vulkan's is.** A roughly constant 2.4–3.0 % offset across both tiers is what
  a less accurate kernel looks like, and it matches the 286× worse per-forward
  floor already measured (8.717e-04 against ROCm's 3.050e-06 at f32).
- So the two GPUs fail for **different reasons**, and Phase 1 should not expect
  one explanation to cover both.

**What "resolution" is confounded with, and must not be glossed over.** The 1024
run is not the 512 pipeline at a finer grid: it adds the whole cascade — the
scaffold upsample, a second flow model, the 1024 conditioning, and one more
decoder subdivision level, plus the chunking that only engages at the larger
row counts. Any of those could carry ROCm's drift. The measurement narrows the
search from "everything" to "what the cascade adds", which is worth a lot; it
does not name the culprit.

### The device is recorded again

These runs are also the first to carry a real device string end to end —
`AMD Radeon AI PRO R9700 [ROCm0]`, `... [Vulkan0]`, `CPU` — surviving a server
restart. That is the ordering fix below working, and it is what makes a table
like the one above reconstructable at all rather than a matter of remembering
which window produced which mesh.

**And the reference is last on every mesh counter.** More boundary edges than
all three backends, more non-manifold edges than all three, with ROCm 6.6×
"cleaner" on boundary edges. This is D1 stated as a measurement rather than an
argument: rank the backends by edge counts and the reference implementation
comes last, which is plainly not a ranking of correctness. Any future claim of
the form "backend X produces a better mesh" has to survive this table first.

Incidental: the finest-level expansion at 1024 is 4.059, inside the healthy
band. The 4.484 false positive seen at 512 does not recur, so the upper
threshold is sound at the resolution it was calibrated for.

Cost breakdown against our own ~203 s for the 1024 geometry path on ROCm: the
reference spends 761 s in the CPU decode and 198 s in the scaffold upsample —
both our own naive fallback conv, not PyTorch. The two flow samplers, where
PyTorch runs its own GPU kernels, are ~5× slower than ours and that ratio still
includes the reference harness's deliberate handicaps (fp32, TF32 off,
query-chunked math SDPA instead of flash attention).

### One bug found on the way

Persisted generations came back from a server restart with no device on their
tile. `server/main.go` wrote the manifest in `persistJob(j)` and only assigned
`j.Device = backendUsed` afterwards, so the in-memory job carried it (visible
until restart) and the manifest never did — the field is `omitempty`, so the key
was absent entirely. The same read also sat *after* `unloadModelsIfIdle()`, so
with `-unload-idle` it could have come back empty even for the live job.

Fixed by capturing the backend at the first progress event — where the
`running on …` log line already reads it, and where the models are certainly
loaded — and assigning it in the same lock as `j.mesh`, before persisting. The
two affected generations were backfilled by hand from the reviewer's account of
which run was which, corroborated by their vertex counts matching the earlier
measurements exactly; they are labelled as reconstructed, because the card
description string the library would have written only ever existed in the
console log.

**It is not yet a defect, and must not be reported as one.** The seed does not
transfer: our RNG is not PyTorch's, so "seed 42" selects a different noise
draw, and two samples from the same distribution can legitimately differ by
this much. Separating "different sample" from "systematically less surface"
needs the tap comparison of Phase 1 and the voxel-set comparison of Phase 1b —
which is exactly the ordering the plan's D1 argues for. What this phase
delivers is the artefact those phases are measured against, plus a number large
enough to be worth measuring.

One incidental observation, recorded because it bears on Phase 2: the
reference's own finest-level expansion here is **4.484** (205,316 → 920,630),
which is above the 4.2 runaway threshold our decoder warns at. Its mesh is not
torn (0.98 % boundary edges, 2.06 tris/vertex). So the upper threshold is
calibrated against our 1024 runs and does not transfer unchanged to 512, or to
the reference. It stays as it is for now — it warns and nothing more — but it
is a known false positive rather than an unexplained one.

## Phase 2 — two-sided diagnostics

D4 in the plan. Two changes, both diagnostic only: no result changes, a broken
one becomes visible.

**`src/trellis2.cpp`** — the finest-level expansion check now also warns below
3.5. The failure it was blind to: a Vulkan run at 1536 collapsed to 163,318
triangles over 2,183,769 vertices with 16.06 % boundary edges, and its
expansion was **2.82** — comfortably under four, so the one-sided check logged
nothing for the worst result this project has recorded. Losing voxels is as
diagnostic as adding them.

**`src/trellis2_capi.cpp`** — a `tris/verts < 1` warning after extraction, not
gated on `TRELLIS2_TIMING`. A dual-grid surface carries roughly two triangles
per vertex; the collapsed run produced 0.075. It is cheaper than the edge walk
and independent of the decoder-side check, which matters precisely because the
decoder-side check did not fire for that run.

Verified against the two recorded failures by their numbers: 2.82 trips the new
lower bound, 0.075 trips the ratio warning, and the healthy values on record
(4.00/4.01/4.04/4.04/3.997/4.008 expansion, 2.39/2.06/2.01 tris per vertex) sit
inside both bands. The 4.484 above is the one value in the corpus that trips a
threshold without a bad mesh behind it.

## Tests

`ctest --test-dir build -C Release -LE model` — **10/10 passed, 0 failed, 0
skipped**, 21.65 s. Nothing in either phase changes a numeric path: the
diagnostics only print, and the new CLI and scripts are additive.

One operational finding worth keeping, now in `AGENTS.md`: on the Windows/HIP
shared-library build, ctest needs the ROCm runtime on `PATH`. Without it, every
test that links `trellis2.dll` **blocks in the loader** rather than failing —
the first run here sat on `test_mesh_export` for 25 minutes at 0 % CPU with an
empty log, which looks exactly like a hung test and is not one. With
`C:\Program Files\AMD\ROCm\6.4\bin` prepended the whole suite is 21 seconds.

`examples/dual_grid_cli` was smoke-tested on a synthetic banded sphere before
the real run, to separate I/O and format bugs from anything the reference
produced, and its output validated against `readMeshFile`'s exact size formula.

## Not done here

Phases 0, 1, 1b and 3 of the plan are untouched. Phase 1b in particular still
needs the full cascade dump (`rocm-native-reference` phase 3) — the dump on
disk was written with `--skip-hr-sampler` and has no `hr_slat`.

Phase 1c was done at 512, as the plan directs. It cost 436 s end to end, of
which 210 s was the CPU decode at 920k voxels; a 1024 run decodes roughly three
million, so the same route at 1024 is plausible in the tens of minutes rather
than the hours the plan feared. That is now a measured estimate instead of an
unknown.
