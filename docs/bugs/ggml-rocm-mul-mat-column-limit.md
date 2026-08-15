# ggml_mul_mat on ROCm/HIP silently stops writing past a fixed column count

Draft for an upstream ggml issue. The behaviour is measured; the explanation at
the bottom is inferred and is flagged as such.

## Summary

On the ROCm/HIP backend, `ggml_mul_mat` stops writing its destination past a
fixed number of columns (`src1->ne[1]`). Everything beyond stays at whatever the
destination buffer held — zero for a freshly allocated graph tensor. The graph
still returns `GGML_STATUS_SUCCESS`, so nothing downstream can tell.

The ceiling depends on the *weight* (`src0`) type:

| `src0` type | last column written |
|-------------|---------------------|
| f16         | 2^21 = 2 097 152    |
| bf16        | 2^21 = 2 097 152    |
| f32         | 2^19 = 524 288      |

It is a fixed threshold, not a fraction of the matrix: a matmul with exactly
2 097 152 columns is correct, one with 2 097 153 columns loses exactly the last
row of output.

### 2026-08-15: this is not ggml. It is the BLAS underneath.

Reproduced with **PyTorch alone, no ggml involved** —
`tools/rocblas_col_truncation.py`. `torch.matmul` on the same card leaves the
tail of its output as exact zeros, silently, with the same signature:

| dtype | L=600k | L=2.2M | L=4.5M | L=9M |
|---|---|---|---|---|
| f32 | **524,288** | **524,288** | **524,288** | **524,288** |
| f16 | ok | ok | **4,194,304** | **8,388,608** |
| bf16 | ok | ok | **4,194,304** | **8,388,608** |
| f64 | ok | — | — | — |

Consequences for the framing above:

- **The title is wrong.** Any ROCm consumer that multiplies a tall, narrow
  matrix is exposed, ggml merely happened to be where we saw it.
- **ROCm 7.2 does not fix it.** That PyTorch run was on `torch.version.hip`
  7.2.26024, a release newer than the 6.4 this project builds against, and the
  f32 cap sits at exactly the same 2^19. An SDK upgrade is not the way out.
- **The thresholds are per kernel, not per backend.** ggml loses f16 at 2^21
  where PyTorch survives to 2^22, so the two dispatch to different kernels and
  each has its own cliff. The f32 cap at 2^19 is identical in both, which is
  what points at a shared layer.
- **It is shape-dependent.** `K = M = 256` is clean where `K = M = 64`
  truncates, so only the narrow-GEMM kernel is affected.

**Inferred, not verified:** 2^19 = 65536 × 8 and 2^22 = 65536 × 64 exactly,
which looks like a grid dimension pinned at 65535 multiplied by the kernel's
tile height rather than an arithmetic overflow. Worth checking first.

### 2026-08-15: Vulkan on the same GPU does not truncate

The strongest evidence, because it removes the hardware from suspicion.
`tests/test_large_rows` built twice from the *same* ggml commit (`30bf8685`),
run on the *same* Radeon AI PRO R9700, differing only in backend:

| op | HIP / ROCm 6.4 | Vulkan (AMD proprietary driver, KHR_coopmat) |
|---|---|---|
| `mul_mat f32` | first bad row **524,288** = 2^19 | **no break** |
| `mul_mat f16` | first bad row **2,097,152** = 2^21 | no break |
| `mul_mat bf16` | first bad row **2,097,152** = 2^21 | **no break** |
| `get_rows`, `add`, `norm`, `silu`, `repeat` | OK | OK |

Note the two failure modes are not the same kind of thing. Vulkan's f16 differs
from the CPU by ~2.6e-03 spread across the whole matrix from row 19 onwards —
ordinary reduced-precision accumulation. HIP's f16 is *bit-identical* to the CPU
up to 2^21 and then drops to exact zeros. One computes inaccurately, the other
stops computing, and only the second leaves no trace.

That closes the chain:

- CPU is correct → not the port
- PyTorch on ROCm 7.2 fails identically → not ggml
- **Vulkan on the same card is correct → not the hardware**

What is left is AMD's BLAS stack (rocBLAS / hipBLASLt / Tensile). The PyTorch
reproducer is short enough to attach to a ticket as it stands; this table is the
argument that the ticket cannot be handed back as a hardware limitation.

### Vulkan names the limit out loud, which settles the mechanism

Running a real 1024³ decode with `TRELLIS2_NO_CHUNK=1` on the Vulkan build —
2,940,217 rows in one graph — does not truncate. It aborts:

```text
ggml-vulkan.cpp:8131: GGML_ASSERT(wg0 <= ctx->device->properties.limits.maxComputeWorkGroupCount[0]
                               && wg1 <= ... && wg2 <= ...) failed
```

So both backends carry the same kind of ceiling — a **compute workgroup count**
limit, which is exactly the "grid dimension pinned at 65535 times the tile
height" the powers of two suggested. That part is no longer inferred.

The difference is what each does at the boundary:

| | at the limit |
|---|---|
| Vulkan | asserts, names the limit, stops |
| HIP / ROCm | writes nothing past it, returns `GGML_STATUS_SUCCESS` |

That reframes the ticket. The complaint is not "the limit is too low" — a limit
is legitimate. It is that **rocBLAS does not check it**, so the caller cannot
tell a completed GEMM from a half-written one. Vulkan demonstrates the correct
behaviour on the same hardware, one assert away.

(Vulkan's own ceiling sits somewhere between 2.2M rows, which `test_large_rows`
clears cleanly, and the 2.94M above. It was never measured precisely, because a
backend that fails loudly does not need a workaround.)

### What was still not measured

- NVIDIA/CUDA was not tested at all. The title says ROCm/HIP because that is
  where it was observed, not because CUDA was checked and found healthy.
- Quantized `src0` types were not tested; they take a different kernel path.
- The exact boundary was found by sampling powers of two, not by bisection, so
  the true cliff could sit anywhere between the last clean and first bad size.

## Environment

- GPU: AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GiB
- ROCm 6.4, Windows 10, clang/clang++ from the ROCm toolchain
- ggml `331b9cba52b23d895bc4ad218c007eb5e667540f`
- `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201`

## Reproduction

Multiply a `[64, 64]` weight by a `[64, L]` activation matrix on the GPU and on
the CPU with identical inputs, then compare:

```
y = ggml_mul_mat(ctx, w /* [64, 64] */, x /* [64, L] */);   // -> [64, L]
```

With `L = 2 200 000` and f16 `w`, GPU and CPU agree exactly up to output row
2 097 151 and then diverge; on the GPU every value from row 2 097 152 on is `0`.
bf16 `w` behaves identically. With f32 `w` the same happens from row 524 288.

For contrast, in the same harness `get_rows`, `mul` (with a broadcast mask),
`add`, `norm`, `silu` and `repeat` all agree over the full 2.2M rows, so the
column count alone is not the problem.

A harness is in `tests/test_large_rows.cpp` of this repository. It takes `L` on
the command line and prints the first diverging row per op. It uses nothing but
`ggml.h`, `ggml-alloc.h`, `ggml-backend.h` and `ggml-cpu.h`, needs no model
files, and is one translation unit, so it can be dropped into ggml's own test
tree unchanged.

```
  get_rows         OK        (max|d| = 0, tol 0)
  mul(mask bcast)  OK        (max|d| = 0, tol 0)
  mul_mat f32      DIVERGES  first bad row 524288  == 2^19 <<<
  mul_mat f16      DIVERGES  first bad row 2097152  == 2^21 <<<
  mul_mat bf16     DIVERGES  first bad row 2097152  == 2^21 <<<
  add              OK        (max|d| = 0, tol 0)
  norm             OK        (max|d| = 7.15e-07, tol 0.0001)
  silu             OK        (max|d| = 5.96e-08, tol 0.0001)
  repeat           OK        (max|d| = 0, tol 0)
```

## Why it matters

Silence is the expensive part. This surfaced in a sparse 3D VAE decoder whose
finest level carries ~2.9M voxels in a `[64, L]` matrix. Its submanifold
convolution is 27 matmuls, so every voxel past 2^21 came out as the convolution
bias alone. The dual-grid mesher then found no isosurface crossings there and
dropped every face in that region — roughly a quarter of the model, sheared off
along a flat plane. Nothing reported an error at any point; the first hint was
the rendered mesh.

The same decoder at half the resolution stays under the limit and is perfectly
fine, so the failure presents as "high resolution is broken" and points the
investigation at the application rather than at the backend. It took an
op-by-op GPU/CPU comparison to land on `mul_mat`.

## Suggested handling

Chunking `src1` along its columns inside `ggml_cuda_mul_mat` would confine the
fix to the backend and keep every caller unaware. Failing that, an explicit
error beats a wrong answer — the current behaviour is indistinguishable from a
correct result to anything except a reference implementation.

Evidence that the chunking direction is sound: we worked around it caller-side
by evaluating that decoder level in blocks of 2^18 voxels, so no single matmul
exceeds the ceiling. Against the unchunked path the block-wise result is
**bit-identical** (max|d| = 0 over 3 072 000 values at 512 000 rows, both at the
production block size and at a deliberately ragged 997-row block), an entire
generation is byte-identical end to end, and the wall-clock difference is within
noise (284.7 s against 284.0 s). Splitting the column dimension therefore costs
nothing measurable and changes no results.

## Speculation, not verified

The two ceilings are `65536 * 32` (f16) and `65536 * 8` (f32). That pattern fits
a GEMM mapping its n dimension onto a grid dimension capped at 65536 blocks,
with a per-block column tile of 32 and 8 respectively. This was inferred from
the ratio between the two measurements alone; the rocBLAS/hipBLAS source was not
consulted, and the actual mechanism may differ.
