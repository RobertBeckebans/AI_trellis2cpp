# Reference environment — producing the PyTorch dumps without Docker

The parity suite compares the C++ port tap-by-tap against activations dumped
from PyTorch (`scripts/dump_*_reference.py`, `tests/ref_*.py`). Those dumps have
to come from somewhere. There are two ways to get them:

| | `docker/Dockerfile.ref` | this document |
|---|---|---|
| backend | CUDA only (`pytorch/pytorch:2.7.1-cuda12.8`) | whatever the host has — ROCm, CUDA, or CPU |
| setup | `docker build` | `uv sync --extra <backend>` |
| status | the documented, reproducible baseline | second path, added so an AMD box can produce dumps at all |

The container stays the baseline. This path exists because the primary
development machine has a **Radeon AI PRO R9700**, which the CUDA image cannot
use — everything there would fall back to CPU, and the cascade dump runs a
12-step sampler over ~40k tokens, which is minutes on a GPU and hours on a CPU.

ROCm PyTorch presents itself as device `cuda`, so none of the dump scripts need
a change: `--device cuda` is correct on both.

> **Read [`docs/plan/rocm-native-reference.md`](plan/rocm-native-reference.md),
> decision D1, before trusting a number produced here.** A reference generated
> on the same backend as the port cannot expose a shared backend defect — the
> test goes green while both sides are wrong. Integer/structural captures are
> unaffected; tight numeric taps are not. Rows in
> [`VERIFICATION.md`](VERIFICATION.md) must name the backend they came from.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and CPython 3.12 (AMD publishes the
Windows ROCm wheels for 3.12 only; `.python-version` pins it).

```sh
uv sync --extra rocm      # AMD, Windows or Linux
uv sync --extra cuda      # NVIDIA
uv sync --extra cpu       # no GPU
```

The package set lives in the repository root [`pyproject.toml`](../pyproject.toml)
and mirrors `docker/Dockerfile.ref`'s pip set (`transformers==4.57.1`,
`safetensors`, `pillow`, `numpy`, `easydict`, `opencv-python-headless`,
`trimesh`, `tqdm`, `imageio`, `gguf`); the extras add the matching torch build.
Only `.venv/` and `uv.lock` are gitignored, so the dependency set itself is
versioned and repeatable.

Backend-specific notes:

- **Windows/ROCm** pulls torch, torchvision and the ROCm runtime (`rocm`,
  `rocm_sdk_core`, `rocm_sdk_libraries_custom`) from
  `repo.radeon.com/rocm/windows/rocm-rel-7.2` — not from PyPI and not from the
  PyTorch index. AMD's Windows torch is built without distributed support, so
  `transformers` has to stay below 5.x, which imports `torch.distributed.fsdp`
  at module level. The 4.57.1 pin from the Dockerfile already satisfies this.
- **Linux/ROCm** uses the PyTorch ROCm index and additionally needs
  `triton-rocm`, which the Linux ROCm torch wheels hard-depend on and which
  exists only on that index.

The reference `trellis2` package is not on PyPI and is not vendored here (it is
gitignored). Point `TRELLIS2_PY` at the checkout that contains `trellis2/`:

```sh
# Windows (PowerShell)
$env:TRELLIS2_PY = "F:\AITools\GFX\AI_trellis2cpp\docs\ideas\TRELLIS-2_rocm"
# Linux
export TRELLIS2_PY=$HOME/python/TRELLIS.2
```

`scripts/ref_common.py` defaults it to `/trellis2`, the container's mount point.

Model weights (~7 GB) go to `models/`:

```sh
scripts/download_models.sh
```

## Verify

```sh
uv run --extra rocm python scripts/ref_env_check.py
```

This is the gate before producing anything. It reports the backend, checks that
`trellis2` imports, and — most importantly — **measures** whether fp32 arithmetic
on the GPU is really fp32, by comparing matmul, SDPA and conv3d against a float64
ground truth. Reading the precision flags is not enough: a flag can be settable
and ignored. Rule of thumb for the reported rel-L2:

- `~1e-7 .. 1e-6` — true fp32, the reference is trustworthy
- `~1e-3` — reduced precision (TF32/xf32-class), it is not

Measured on the R9700 (gfx1201, torch 2.9.1+rocmsdk20260116, 2026-08-14):

```text
                      cpu        cuda default   cuda after setup
matmul                2.3e-07    1.15e-06       1.15e-06
sdpa                  3.7e-07    5.10e-07       5.01e-07
conv3d                1.9e-07    3.62e-07       3.62e-07
```

## Precision: what `ref_common.setup()` does on ROCm

`_force_true_fp32()` exists because PyTorch's CUDA defaults cost about 10 bits
of mantissa, and that trap once made a correct port look like it had a
0.08-rel-L2 bug (see [`VERIFICATION.md`](VERIFICATION.md)). Its switches are
named after CUDA features, so whether they mean anything on ROCm was an open
question — decision D4 of the plan. Measured answer:

**The switches are not silently discarded — they change state:**

| switch | ROCm default | after `_force_true_fp32()` |
|---|---|---|
| `cuda.matmul.allow_tf32` | `False` | `False` |
| `cudnn.allow_tf32` | `True` | `False` |
| `cuda.matmul.fp32_precision` | `'none'` | `'ieee'` |
| `cudnn.conv.fp32_precision` | **`'tf32'`** | `'none'` |
| `cuda.flash_sdp_enabled()` | `True` | `False` |
| `cuda.mem_efficient_sdp_enabled()` | `True` | `False` |
| `cuda.math_sdp_enabled()` | `True` | `True` |

**But on this GPU they change no numbers.** All three ops measure at true fp32
*before* the switches are applied as well as after. Most striking:
`cudnn.conv.fp32_precision` reports `'tf32'` by default, yet conv3d still comes
out at 3.6e-07 — RDNA4 has no TF32-class reduced-precision matrix path for fp32,
so the setting is advisory and the hardware runs full width regardless.

Two consequences, and the second is the one that will bite someone:

1. **On gfx1201, numeric taps are at reference precision.** The precision half
   of D4 is closed. What remains against the numeric taps is D1 — shared-backend
   cancellation — which is a property of the method, not of the arithmetic, and
   no measurement can clear it.
2. **Do not generalise this to other AMD hardware.** CDNA (MI300, gfx942) does
   have an xf32 path, where that default `'tf32'` conv setting would really cost
   mantissa. Rerun `ref_env_check.py` on any new card and believe the measured
   numbers, not this table.

`setup()` also monkeypatches sparse attention to dense SDPA and registers a
pure-PyTorch submanifold sparse-conv backend, so no custom CUDA/HIP extensions
(flash-attn, FlexGEMM, cumesh, o-voxel) are needed. That is unchanged here.

## Producing dumps

The container orchestration is `scripts/refgen.sh`. Natively the same sequence
is, from the repository root with `TRELLIS2_PY` set:

```sh
FIXTURE=$TRELLIS2_PY/assets/example_image/0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp

uv run --extra rocm python scripts/dump_dino_reference.py --image "$FIXTURE"
uv run --extra rocm python tests/ref_ss_flow.py                  # CPU by design: true fp32 golden
uv run --extra rocm python tests/ref_ss_sample.py  --device cuda
uv run --extra rocm python tests/ref_ss_dec.py     --device cuda
uv run --extra rocm python scripts/dump_slat_reference.py    --device cuda
uv run --extra rocm python scripts/dump_cascade_reference.py --device cuda
```

The GGUF converters (`scripts/convert_*_to_gguf.py`) run in the same environment
and need no GPU.
