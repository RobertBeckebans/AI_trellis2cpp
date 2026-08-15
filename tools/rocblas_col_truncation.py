#!/usr/bin/env python3
"""rocBLAS/hipBLAS on gfx1201 silently truncates tall GEMMs — PyTorch only.

Self-contained reproducer for the defect that forced the decoder-side chunking
in src/trellis2.cpp. Deliberately uses nothing but PyTorch: the point is that
this is not a ggml bug. `torch.matmul` returns a result whose tail rows are
left as exact zeros, with no error and no warning.

Measured 2026-08-15 on a Radeon AI PRO R9700 (gfx1201), Windows 10,
torch 2.9.1+rocmsdk20260116, torch.version.hip 7.2.26024 — i.e. ROCm 7.2, one
release newer than the 6.4 the C++ build links against, and still broken:

    dtype     L=600k     L=2.2M     L=4.5M       L=9M
    f32       524,288    524,288    524,288      524,288     <- hard cap at 2^19
    f16       ok         ok         4,194,304    8,388,608
    bf16      ok         ok         4,194,304    8,388,608
    f64       ok         -          -            -

Two observations that should help whoever fixes it:

  * the f32 cap is exactly 2^19 = 65536 * 8 and the f16 one exactly
    2^22 = 65536 * 64, which looks like a grid dimension pinned at 65535 times
    the kernel's tile height rather than an arithmetic overflow;
  * it disappears at larger K/M (K=M=256 is clean where K=M=64 truncates), so a
    different kernel is selected there and only the narrow-GEMM path is affected.

Why it matters: the rows are not merely inaccurate, they are never written. Any
workload that multiplies a tall, narrow matrix — per-element or per-voxel
networks, large batches of small features — gets silent zeros for everything
past the cliff. In this project it cost every mesh face past 2^21 voxels in a
1024³ decode, which is what the chunking now works around.

Usage:  uv run --extra rocm python tools/rocblas_col_truncation.py
Exit:   0 clean, 1 truncation observed, 2 no GPU.
"""

import sys

import torch

CI = CO = 64        # narrow GEMM: this is the shape that triggers it
CHUNK = 131_072     # reference is the same product in pieces below every cliff


def last_written_row(a, b):
    """Rows of a@b that the GEMM actually touched (a zero tail means truncated)."""
    out = a @ b
    nonzero = (out.float().abs().sum(dim=1) != 0)
    n = int(nonzero.nonzero()[-1].item()) + 1 if nonzero.any() else 0
    del out, nonzero
    torch.cuda.empty_cache()
    return n


def main():
    if not torch.cuda.is_available():
        print("no GPU visible")
        return 2

    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"torch  : {torch.__version__}")
    print(f"HIP    : {getattr(torch.version, 'hip', None)}")
    print(f"CUDA   : {getattr(torch.version, 'cuda', None)}")
    print(f"\n2^19 = {1 << 19:,}   2^21 = {1 << 21:,}   2^22 = {1 << 22:,}\n")

    truncated = False

    print(f"=== last written row, K=M={CI} ===")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        for L in (600_000, 2_200_000, 4_500_000, 9_000_000):
            g = torch.Generator(device="cpu").manual_seed(0)
            a = torch.randn(L, CI, generator=g).to("cuda", dtype)
            b = torch.randn(CI, CO, generator=g).to("cuda", dtype)
            n = last_written_row(a, b)
            del a, b
            torch.cuda.empty_cache()
            if n == L:
                print(f"  {name:9s} L={L:>10,}  all written")
            else:
                truncated = True
                pow2 = f"  (2^{n.bit_length() - 1})" if n and (n & (n - 1)) == 0 else ""
                print(f"  {name:9s} L={L:>10,}  TRUNCATED at {n:>10,}{pow2}")
        print()

    # The tail is not wrong-but-close, it is untouched: show that the head is
    # bit-identical to the chunked product and the tail is exactly zero.
    print("=== f32, L=600,000: is the head exact and the tail exactly zero? ===")
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(600_000, CI, generator=g).to("cuda")
    b = torch.randn(CI, CO, generator=g).to("cuda")
    full = a @ b
    ref = torch.cat([a[s:s + CHUNK] @ b for s in range(0, 600_000, CHUNK)])
    cut = 1 << 19
    print(f"  rows <  2^19 identical to the chunked product : {torch.equal(full[:cut], ref[:cut])}")
    print(f"  rows >= 2^19 all exactly zero                 : {bool((full[cut:] == 0).all().item())}")
    print(f"  mean |correct value| in that range            : {ref[cut:].abs().mean().item():.3f}")
    del a, b, full, ref
    torch.cuda.empty_cache()

    print("\n=== does a wider GEMM dodge it? (f32, L=600,000) ===")
    for k in (16, 64, 256):
        g = torch.Generator(device="cpu").manual_seed(0)
        a = torch.randn(600_000, k, generator=g).to("cuda")
        b = torch.randn(k, k, generator=g).to("cuda")
        n = last_written_row(a, b)
        del a, b
        torch.cuda.empty_cache()
        print(f"  K=M={k:4d} -> last written {n:>9,}{'' if n == 600_000 else '   TRUNCATED'}")

    print("\nverdict:", "TRUNCATION OBSERVED" if truncated else "clean on this host")
    return 1 if truncated else 0


if __name__ == "__main__":
    sys.exit(main())
