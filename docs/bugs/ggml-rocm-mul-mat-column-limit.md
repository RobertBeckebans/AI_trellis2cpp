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
| f32         | 2^19 = 524 288      |

It is a fixed threshold, not a fraction of the matrix: a matmul with exactly
2 097 152 columns is correct, one with 2 097 153 columns loses exactly the last
row of output.

### What was not measured

- Only one shape was tested: `src0` `[64, 64]`, i.e. M = K = 64. Whether the
  ceiling moves with M or K is unknown, so the two numbers above may be specific
  to this shape rather than universal.
- NVIDIA/CUDA was not tested at all. The title says ROCm/HIP because that is
  where it was observed, not because CUDA was checked and found healthy.
- bf16 `src0` was not tested; only f16 and f32.

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
With f32 `w` the same happens from row 524 288.

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
