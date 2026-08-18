# Progress — the Apple Silicon port as a second reference, and what comparing it found

- **Issue:** none — reference-provenance work
- **Plans:** [`backend-parity`](../plan/backend-parity.md) D2b,
  [`rocm-native-reference`](../plan/rocm-native-reference.md) D1
- **Branch:** `pixal3d`
- **Date:** 2026-08-17
- **Commits:** this one

## Goal

A second independent fork of TRELLIS.2 arrived (`docs/ref/trellis2-apple/`, MLX /
Apple Silicon). Two forks of one upstream, hitting two different walls, is a
cheap way to separate "this is how TRELLIS.2 works" from "this is what one
platform needed" — which is exactly the question `backend-parity` D2b exists to
answer.

## Outcome in one line

The two forks agree on 71 of 94 shared files; of the 23 that differ, the one that
matters is a DINOv3 branch in the ROCm fork that **silently changes every
reference dump** the moment `transformers` is upgraded to 5.x.

## Shape of the comparison

Both packages carry 96 Python files.

| | count |
|---|---|
| shared and **byte-identical** | **71** |
| shared and differing | 23 |
| only in the ROCm fork | 2 — `modules/sparse/conv/conv_none.py`, `utils/pipeline_logger.py` |
| only in the Apple port | 2 — `backends.py`, `utils/grid_sample.py` |

The differences split cleanly by direction: the ROCm fork *adds* workarounds, the
Apple port *removes* CUDA assumptions. Neither derives from the other.

(`conv_pytorch.py` counts as shared because a copy was placed in `trellis2/`
during this work. It is byte-identical to the Apple original and currently inert,
because `ref_common` overrides the backend selector — see D2b.)

## The finding that matters: a version-dependent DINOv3 path

`trellis2/modules/image_feature_extractor.py:97` branches on
`hasattr(self.model, 'model')`:

- **Taken** → `self.model(image).last_hidden_state`, which in HF is
  `self.norm(hidden_states)` (`modeling_dinov3_vit.py:529`) — a LayerNorm **with**
  affine parameters — over which the fork then applies a second, affine-free
  `F.layer_norm`.
- **Not taken** → the manual layer loop, which never touches `self.norm` at all.

These are different features. Nothing errors either way.

With the installed **transformers 4.57.1** the branch is *not* taken:
`DINOv3ViTModel` declares `self.layer` at the top level
(`modeling_dinov3_vit.py:493`) and has no `.model` attribute. So today's dumps
take the manual path, and `test_dino`'s 7e-7 is honest.

On transformers 5.x the attribute nests one level deeper, the branch fires, and
every DINO reference moves — while `test_dino` reports a *port* failure. That is
the identical trap to the `out7` episode
([`rocm-native-reference_3-slat-dump-and-out7.md`](rocm-native-reference_3-slat-dump-and-out7.md)):
a reference-side defect wearing a port defect's clothes, which sends the search
to the side that has nothing to find.

The Apple port does not have it. Its `_encoder` property resolves by looking for
`.layer` — first on the model, then on `.model` — and iterates manually in both
layouts, so no library version selects a different mathematical path.

## `ref_common`'s convolution: a better suspect, from the fork's own code

`scripts/ref_common.py` registers its own sparse convolution and is wrong on
ROCm (4.9x too large, uncorrelated, not reproducible run to run). The suspect
recorded at the time was the masked scatter-add `out[hit] += contrib`.

The ROCm fork supplies a better one. `trellis2/modules/sparse/linear.py:11`:

```python
# ROCm GFX1201 (RX 9070 XT) bug workaround:
# hipBLASLt and rocBLAS GEMM kernels corrupt memory (-> NaN) when N > ~800k
# for shapes like [N, K] @ [K, M] with small K/M.  Chunking keeps each
# dispatch below the confirmed-safe threshold of 524288 rows.
ROCM_SAFE_CHUNK = 524_288
```

The fork honours that line in three places (`SparseLinear`, the ConvNeXt MLP, and
an optional chunked spconv). `ref_common.py:243` does not: `feats[src] @ w.t()`
is exactly the named shape — `[N, K] @ [K, M]` with small K/M — and runs
unchunked to 1.19M rows.

The level boundary fits without remainder:

| step | ≈ rows | vs. 524,288 | measured |
|---|---|---|---|
| levels 0–3 | ≤ ~296k | under | PASS, ≤ 1.1e-06 |
| `conv1` (before `updown`) | ~296k | under | `in_hch` PASS, 3.681e-07 |
| `norm2` (no GEMM) | 1.19M | — | PASS, cos +1.000000 |
| **`conv2`** | **1.19M** | **over** | **FAIL, cos +0.164** |

The scatter-add explains the amplitude but not why the break lands exactly where
the first GEMM crosses that row count. This does. Still unconfirmed — chunking
`ref_common`'s conv and regenerating on ROCm is the one-line test.

**And there is a ready replacement.**
`docs/ref/trellis2-apple/trellis2/modules/sparse/conv/conv_pytorch.py` is pure
PyTorch, device-agnostic, and algorithmically what this port does: hash →
neighbour map → gather → `bmm` → `sum` over the 27 offsets. No masked
scatter-add, no atomics, deterministic anywhere.

Two caveats before it becomes the reference. It shares an algorithm shape with
the port, so it is *less* independent than an alien implementation — though
`ref_common`'s own conv was written for this project too, so this is a step
sideways rather than backwards. And it neither chunks nor bounds its
`batch·D·H·W` int64 lookup table, which at 1024³ is 8.6 GB on its own.

## Two ROCm workarounds that do nothing

Both would send a precision investigation the wrong way.

**`convert_module_to_bf16` is never called.** It is defined
(`modules/utils.py:28`) and imported (`sparse_unet_vae.py:6`) and that is all —
both `convert_to_fp16` docstrings claim "actually bfloat16 for ROCm stability"
while the code applies `convert_module_to_f16`.

**`subdiv.feats.float() > 0`** (`sparse_unet_vae.py:248`), commented "bf16 trained
weights produce shifted logits", cannot change any comparison: bf16 → f32 is
lossless and preserves sign exactly.

## Three upstream bugs the Apple port fixed and the ROCm fork still carries

None of them are Apple-specific.

| Where | Defect |
|---|---|
| `pipelines/__init__.py` | `globals()[config['name']]` bypasses the module's lazy `__getattr__` and raises `KeyError` when `from_pretrained` is the first touch. Fix: `getattr(sys.modules[__name__], ...)` |
| `pipelines/trellis2_texturing.py` | `trimesh.Trimesh(vertices, faces, process=False)` drops `mesh.visual`, making `postprocess_mesh`'s keep-existing-UVs branch unreachable; and `mesh.vertex_normals` is a read-only trimesh cache that the axis swap then writes in place. Fix: carry `visual=`, and `np.array(...)` the three arrays |
| `modules/sparse/attention/windowed_attn.py` | the xformers cross-attention branch passes `q` where it means `q_feats`. Only reachable with xformers selected |

## What is only in the Apple port

- **`mlx_backend/`** (3,270 lines) — an independent reimplementation of DINOv3,
  the flow models, sparse conv, the VAE decoders and the structure decoder. Not
  runnable here, but a second witness whenever a tensor layout or a norm
  placement is in question.
- **`o-voxel/`** in full, including `test_dual_grid_mesh_indices.py`:
  `!= 0xffffffff` on an int32 tensor matches on CPU and CUDA (the out-of-range
  scalar wraps to -1) but not on MPS (compared in a wider type), so every quad
  with a missing corner survived and carried -1 into the faces. Our
  `examples/flexible_dual_grid.h` builds edge keys and has no sentinel, so the
  class is excluded by construction — worth knowing rather than worth porting.
- **`utils/grid_sample.py`** — an `F.grid_sample` fallback. **Not** usable as a
  reference for `test_pbr_sampling`: it fills absent voxels with zero and does no
  sparse-boundary renormalisation, which is precisely what the port implements.

Only in the ROCm fork: `_safe_ssaa` (a render-time VRAM cap), `pipeline_logger`,
the `visualize_*` helpers, and `conv_none.py` — which
[D2b](../plan/backend-parity.md) already records as never running.

## Verification

Everything above is a file comparison, so it is reproducible rather than measured:

```sh
diff -r trellis2 docs/ref/trellis2-apple/trellis2
```

The one runtime check is the transformers layout, confirmed against the installed
4.57.1: `DINOv3ViTModel` has `.layer` and `.norm` at the top level and no
`.model`, so the branch is not taken today.

## Open points

- **`docs/ref/` is untracked.** A whole third-party MIT repo. Until that is
  decided, this document cites paths a reader may not have.
- **The unchunked-GEMM hypothesis is untested.** Chunk `ref_common`'s conv at
  524,288 rows, regenerate the SLAT dump on ROCm, run `test_slat`. If it passes,
  reference dumps cost 4 minutes again instead of 75.
- **The DINOv3 branch is a live trap**, not a historical one. Either adopt the
  Apple port's `_encoder` resolution or pin `transformers` and say why.
- **Licence.** `docs/ref/trellis2-apple/LICENSE` is MIT, Copyright Microsoft — a
  fourth data point for `pixal3d-proj-conditioning` D0, which stays the
  reviewer's call.

## Proposed commit message

```text
Compare the two TRELLIS.2 forks, and find a version-dependent DINOv3 path

The Apple Silicon port is a second independent fork of the same upstream, so
diffing it against the ROCm one separates "how TRELLIS.2 works" from "what one
platform needed". 71 of 94 shared files are byte-identical; the ROCm fork adds
workarounds, the Apple port removes CUDA assumptions, and neither derives from
the other.

The find that matters is image_feature_extractor.py branching on
hasattr(self.model, 'model'). Not taken, it iterates the layers manually. Taken,
it returns last_hidden_state - which is self.norm(...), an AFFINE LayerNorm - and
then applies a second affine-free layer_norm over it. Different features, no
error either way. transformers 4.57.1 does not take it, so today's dumps and
test_dino's 7e-7 are honest; 5.x does, and every DINO reference moves while the
test blames the port. The Apple port resolves the encoder by looking for .layer
instead of by version, so no library upgrade picks a different path.

Also: a better suspect for ref_common's ROCm-wrong convolution, from the fork's
own linear.py - hipBLASLt corrupts [N,K]@[K,M] with small K/M past ~800k rows and
the fork chunks at 524,288 in three places, while ref_common's per-offset GEMM
runs unchunked to 1.19M. The level boundary fits exactly: everything under the
threshold passes, conv2 is the first step over it and is the one that fails.
conv_pytorch.py is a ready device-agnostic replacement.

Two ROCm workarounds are inert and would misdirect a precision hunt:
convert_module_to_bf16 is imported and never called while two docstrings claim
bf16, and subdiv.feats.float() > 0 cannot change a sign. Three upstream bugs the
Apple port fixed are still open in the fork.

Detail in docs/progress/apple-port-comparison.md.
```
