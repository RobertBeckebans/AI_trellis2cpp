#!/usr/bin/env python3
"""Check that this host can produce the PyTorch reference dumps.

The reference dumps used to be produced only in docker/Dockerfile.ref, which is
CUDA-only. This script verifies the native alternative (see
docs/reference-environment.md):

  1. torch sees a GPU (ROCm PyTorch reports itself as device "cuda", so the
     dump scripts need no change),
  2. the `trellis2` reference package is importable via TRELLIS2_PY,
  3. `ref_common.setup()`'s precision switches actually have an effect here.

(3) is the one that matters. The switches in `_force_true_fp32()` are named
after CUDA features (TF32, flash/mem-efficient SDPA). On a non-CUDA backend they
may be silent no-ops, and a "reference" produced at reduced precision is worse
than no reference at all. Reading the flags is not enough — a flag can be
settable and ignored — so this measures the actual arithmetic against a float64
ground truth, once in the backend's default state and once after the switches
are applied.

Interpretation of the measured rel-L2 versus float64:

    ~1e-7 .. 1e-6   true fp32, the reference is trustworthy
    ~1e-3           reduced precision (TF32/xf32-class), the reference is NOT

Usage:
  uv run --extra rocm python scripts/ref_env_check.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_common  # noqa: E402  (sets sys.path for trellis2, installs stubs)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FP32_GATE = 1e-5  # comfortably above true fp32, far below TF32-class error


def rel_l2(got, ref):
    ref = ref.double()
    return ((got.double() - ref).norm() / ref.norm()).item()


def report_backend():
    print("== backend ==")
    print(f"torch              {torch.__version__}")
    print(f"torch.version.hip  {getattr(torch.version, 'hip', None)}")
    print(f"torch.version.cuda {getattr(torch.version, 'cuda', None)}")
    ok = torch.cuda.is_available()
    print(f"cuda.is_available  {ok}")
    if ok:
        print(f"device 0           {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"gcn arch           {getattr(props, 'gcnArchName', 'n/a')}")
        print(f"total memory       {props.total_memory / 2**30:.1f} GiB")
    return ok


def snapshot_switches():
    s = {
        "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    # torch >= 2.9 replaces the booleans with precision strings; on ROCm these
    # are the switches that actually reach the dispatcher.
    for name, obj in (("cuda.matmul.fp32_precision", torch.backends.cuda.matmul),
                      ("cudnn.conv.fp32_precision", getattr(torch.backends.cudnn, "conv", None))):
        if obj is not None and hasattr(obj, "fp32_precision"):
            s[name] = obj.fp32_precision
    for fn in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        if hasattr(torch.backends.cuda, fn):
            s[f"cuda.{fn}()"] = getattr(torch.backends.cuda, fn)()
    return s


def make_inputs():
    g = torch.Generator(device="cpu").manual_seed(0)
    return {
        "matmul": (torch.randn(1024, 4096, generator=g), torch.randn(4096, 1024, generator=g)),
        "sdpa": tuple(torch.randn(1, 8, 512, 64, generator=g) for _ in range(3)),
        # conv3d is its own dispatch path (MIOpen/cuDNN), and it is what the
        # dense SS decoder runs on — cudnn.conv.fp32_precision governs it.
        "conv3d": (torch.randn(1, 16, 24, 24, 24, generator=g),
                   torch.randn(16, 16, 3, 3, 3, generator=g)),
    }


def golds(inp):
    a, b = inp["matmul"]
    q, k, v = inp["sdpa"]
    x, w = inp["conv3d"]
    return {
        "matmul": a.double() @ b.double(),
        "sdpa": F.scaled_dot_product_attention(q.double(), k.double(), v.double()),
        "conv3d": F.conv3d(x.double(), w.double(), padding=1),
    }


def run_ops(inp, device):
    a, b = inp["matmul"]
    q, k, v = inp["sdpa"]
    x, w = inp["conv3d"]
    to = lambda t: t.to(device)  # noqa: E731
    return {
        "matmul": (to(a) @ to(b)).cpu(),
        "sdpa": F.scaled_dot_product_attention(to(q), to(k), to(v)).cpu(),
        "conv3d": F.conv3d(to(x), to(w), padding=1).cpu(),
    }


def measure(inp, gold, device, label):
    print(f"\n== fp32 arithmetic on {device}, {label} (rel-L2 vs float64) ==")
    got = run_ops(inp, device)
    all_ok = True
    for op in ("matmul", "sdpa", "conv3d"):
        err = rel_l2(got[op], gold[op])
        ok = err < FP32_GATE
        all_ok &= ok
        print(f"  {op:8s} {err:.3e}   <- {'true fp32' if ok else 'REDUCED PRECISION'}")
    return all_ok


def report_trellis2():
    print("\n== trellis2 reference package ==")
    print(f"TRELLIS2_PY      {ref_common.TRELLIS2_PY}")
    if not os.path.isdir(os.path.join(ref_common.TRELLIS2_PY, "trellis2")):
        print("  MISSING — set TRELLIS2_PY to the checkout that contains trellis2/")
        return False
    try:
        import trellis2  # noqa: F401
        print(f"  import trellis2  OK ({trellis2.__file__})")
        return True
    except Exception as e:  # noqa: BLE001 — the point is to report, not to raise
        print(f"  import trellis2  FAILED: {type(e).__name__}: {e}")
        return False


def report_assets():
    print("\n== assets ==")
    for label, path in (("models/", ref_common.MODELS), ("dumps/", ref_common.DUMPS)):
        if os.path.isdir(path):
            n = sum(len(f) for _, _, f in os.walk(path))
            print(f"  {label:8s} present ({n} files)")
        else:
            print(f"  {label:8s} absent")


def main():
    have_gpu = report_backend()
    device = "cuda" if have_gpu else "cpu"

    inp = make_inputs()
    gold = golds(inp)
    print("\n== cpu baseline (rel-L2 vs float64) ==")
    for op, err in ((op, rel_l2(v, gold[op])) for op, v in run_ops(inp, "cpu").items()):
        print(f"  {op:8s} {err:.3e}")

    before = snapshot_switches()
    default_ok = measure(inp, gold, device, "backend default") if have_gpu else True

    ref_common._force_true_fp32()
    after = snapshot_switches()
    print("\n== ref_common._force_true_fp32() ==")
    for k in before:
        mark = "->" if before[k] != after[k] else "  "
        print(f"{mark} {k:34s} {before[k]!r:12s} {after[k]!r}")
    if all(before[k] == after[k] for k in before):
        print("   (no flag changed — either already at the target state, or ignored here)")

    forced_ok = measure(inp, gold, device, "after _force_true_fp32()") if have_gpu else True

    report_trellis2()
    report_assets()

    print("\n== verdict ==")
    if not have_gpu:
        print("no GPU — the dump scripts will run on CPU (correct, but slow)")
    elif forced_ok:
        print(f"{device} is at full fp32 after setup; numeric taps produced here")
        print("are at reference precision.")
        if not default_ok:
            print("The switches are NOT no-ops here: the backend default is reduced")
            print("precision for at least one op. Never dump without ref_common.setup().")
    else:
        print(f"{device} is NOT at full fp32 even after setup — do not produce numeric")
        print("taps on it.")
    print("\nnote: precision is only half the story. A reference produced on the")
    print("same backend as the port cannot expose a shared backend defect; see")
    print("docs/plan/rocm-native-reference.md, D1.")


if __name__ == "__main__":
    main()
