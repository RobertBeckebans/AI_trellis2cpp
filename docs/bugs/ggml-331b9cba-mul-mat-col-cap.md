# Local ggml fix for the mul_mat column truncation (fork only)

Applies to the vendored ggml at `331b9cba` (version 0.9.9). **Does not port to
ggml master**, where `ggml_cuda_op_mul_mat` and its `src1_col_stride` loop no
longer exist — `ggml_cuda_mul_mat` now dispatches straight to `mul_mat_vec_f` /
`mul_mat_f` / `mul_mat_vec_q` / `mul_mat_q` and falls through to
`ggml_cuda_mul_mat_cublas`. Kept as a record of what was verified, not as
something to re-apply.

Verified on gfx1201 / ROCm 6.4: with these two hunks `tests/test_large_rows`
reports no divergence for f32, f16 or bf16 weights at L = 2 200 000, where the
unpatched build drops every output row past 2^19 (f32) and 2^21 (f16, bf16).

Both hunks are in `src/ggml-cuda/ggml-cuda.cu`.

## 1. Next to `#define MUL_MAT_SRC1_COL_STRIDE 128`

```c
// Upper bound on the columns handed to one GEMM. rocBLAS on gfx1201 silently
// stops writing its output past 2^21 columns with f16/bf16 weights and past 2^19
// with f32 — no error, the remaining destination rows are simply left untouched.
// The split loop below already exists for multi-device tensor splits, so reusing
// it costs nothing on shapes that stay under the bound, and shapes that large are
// rare outside dense per-element workloads. Left unbounded on CUDA, where the
// truncation has not been observed and the extra iterations would be pointless.
#ifdef GGML_USE_HIP
#define MUL_MAT_SRC1_COL_MAX (1 << 18)
#else
#define MUL_MAT_SRC1_COL_MAX INT64_MAX
#endif
```

## 2. In `ggml_cuda_op_mul_mat`

```c
-    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
+    const int64_t src1_col_stride = split && used_devices > 1
+        ? MUL_MAT_SRC1_COL_STRIDE
+        : std::min<int64_t>(ne11, MUL_MAT_SRC1_COL_MAX);
```

## Scope

Only the `ggml_cuda_op_mul_mat` path is covered. The dispatch has five other
branches whose kernels were not tested at large `ne11`; the MMQ path for
quantized weights in particular could truncate the same way and would not be
helped by this. Our case reaches the patched branch only because the tensor is
2D and the batched-cuBLAS branch requires `ne2*ne3 > 1`.

The decoder-side split in `trellis2.cpp` does not depend on any of this and is
the fix that survives a ggml bump.
