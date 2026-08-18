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

**The `transformers` pin has a second reason, and it outlives the first.**
A 5.x upgrade is a **reference-affecting change** on any platform.
`trellis2/modules/image_feature_extractor.py` branches on
`hasattr(self.model, 'model')`: not taken it iterates the transformer layers by
hand, taken it returns `last_hidden_state` — `self.norm(hidden_states)`, a
LayerNorm *with* affine parameters — and then applies a second, affine-free
`F.layer_norm` over it. Two different feature tensors, and no error either way.
On 4.57.1 the branch is not taken (`DINOv3ViTModel` declares `self.layer` at the
top level and has no `.model`), which is why `dumps/reference_dino.gguf` and
`test_dino` are honest today. On 5.x it fires, every DINO reference moves, and
`test_dino` points at the port — the same shape of trap as the `out7` episode in
`docs/progress/rocm-native-reference_3-slat-dump-and-out7.md`, which sent two
days of searching to the side that had nothing to find.

So the Windows reason is platform-specific and the distributed-support gap will
eventually close; this one will not. If the pin ever has to move, adopt the Apple
port's resolution first (`docs/ref/trellis2-apple/`): its `_encoder` property
looks for `.layer` on the model and then on `.model`, so both layouts iterate
manually and no library version selects a different mathematical path. Then
regenerate the DINO reference and re-measure, in that order.
- **Linux/ROCm** uses the PyTorch ROCm index and additionally needs
  `triton-rocm`, which the Linux ROCm torch wheels hard-depend on and which
  exists only on that index.

The reference `trellis2` package is not on PyPI and is not vendored here. The
simplest arrangement is a copy in the repository root, which is gitignored:

```sh
cp -r docs/ref/TRELLIS-2_rocm/trellis2 ./trellis2
```

`scripts/ref_common.py` picks that up on its own. It is MIT-licensed, so there
is no conflict, but it stays untracked deliberately: it is reference material we
run against, not code this project adopts.

For a checkout somewhere else, point `TRELLIS2_PY` at the directory that
*contains* `trellis2/`:

```sh
# Windows (PowerShell)
$env:TRELLIS2_PY = "F:\AITools\GFX\AI_trellis2cpp\docs\ref\TRELLIS-2_rocm"
# Linux
export TRELLIS2_PY=$HOME/python/TRELLIS.2
```

Without a root copy and without the variable, `ref_common` falls back to
`/trellis2`, the container's mount point.

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

## Producing a reference generation

The dumps above are per-stage activations for the parity tests. A *whole mesh*
from the reference implementation, for side-by-side viewing against the
backends, is three steps:

```sh
uv run --extra rocm python scripts/ref_generate.py \
    --image assets/einstein.png --seed 42 --pipeline-type 512 \
    --out dumps/einstein_ref512.fdgvox --out-json dumps/einstein_ref512.json

./build/bin/Release/dual_grid_cli dumps/einstein_ref512.fdgvox \
                                  dumps/einstein_ref512.t2mesh

uv run --extra rocm python scripts/ref_publish_generation.py \
    --mesh dumps/einstein_ref512.t2mesh --image assets/einstein.png \
    --info dumps/einstein_ref512.json
```

`ref_generate.py` runs the upstream pipeline stages and stops at the shape
decoder's raw 7 channels — the step after that is o_voxel's CUDA mesher, which
does not build on Windows. `dual_grid_cli` meshes those channels with **our**
extractor, the same one every backend uses, so the resulting artefact is
"PyTorch geometry, our extraction" and any difference against a trellis2.cpp
run is the network rather than the mesher. `ref_publish_generation.py` writes
the `generations/<id>/` layout the server restores at startup, labelled
`[PyTorch]` so the viewer's history tile says what it is.

Einstein at seed 42, type `512`, cost 436 s on an R9700 (the decode runs on
CPU, as it does in the port). `--pipeline-type 1024_cascade` runs the HR chain
instead; budget roughly three times the decode.

The seed does **not** transfer between this and a trellis2.cpp run — our RNG is
not PyTorch's — so treat mesh-count differences as a sample difference until a
tap or voxel-set comparison says otherwise.
