#!/usr/bin/env python3
"""
Convert the TRELLIS.2 shape-SLAT flow DiT checkpoint (sparse DiT over active
voxels; identical block structure to the SS-flow model, RoPE over voxel coords)
(slat_flow_img2shape_dit_1_3B_512_bf16.safetensors) to a GGUF file for trellis2.cpp.

This is the stage-1 generator: a ~1.3B-param DiT with adaLN-Zero modulation
(share_mod), self-attention + cross-attention to the DINOv3 cond tokens, 3D
RoPE, and QK-RMSNorm. See trellis2/models/sparse_structure_flow.py.

Like the sam3.cpp converters this is a self-contained script (only safetensors
+ numpy + torch). It writes a standard GGUF v3 file — no `gguf` package needed —
so the C++ side loads it with ggml's built-in gguf_init_from_file(): tensors are
keyed by their original checkpoint names and hyperparameters travel as KV
metadata under the `trellis2.slat_flow.*` namespace.

Usage:
    python convert_ss_flow_to_gguf.py \
        --model /path/to/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
        --output ss_flow_dit.gguf --ftype 1

    # --model/--config default to the microsoft/TRELLIS.2-4B HF cache snapshot.

ftype: 0 = f32 (lossless upcast from the bf16 checkpoint),
       1 = f16 (default; big 2-D weight matrices only, norms/gammas stay f32),
       2 = bf16 (lossless, native checkpoint precision; needs bf16-capable ggml).
"""

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

# ── GGUF / GGML constants (must match the bundled ggml) ──────────────────────
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

# GGUF metadata value types
GGUF_VT_UINT32 = 4
GGUF_VT_INT32 = 5
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL = 7
GGUF_VT_STRING = 8

ARCH = "trellis2-slat-flow"
KV_PREFIX = "trellis2.slat_flow."

DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/*/ckpts"
)
CKPT_STEM = "slat_flow_img2shape_dit_1_3B_512_bf16"


# ── GGUF writer (minimal, v3) ────────────────────────────────────────────────
def _gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _gguf_str(key) + struct.pack("<I", vtype) + payload


def kv_u32(key, v):    return _kv(key, GGUF_VT_UINT32,  struct.pack("<I", int(v)))
def kv_i32(key, v):    return _kv(key, GGUF_VT_INT32,   struct.pack("<i", int(v)))
def kv_f32(key, v):    return _kv(key, GGUF_VT_FLOAT32, struct.pack("<f", float(v)))
def kv_bool(key, v):   return _kv(key, GGUF_VT_BOOL,    struct.pack("<?", bool(v)))
def kv_str(key, v):    return _kv(key, GGUF_VT_STRING,  _gguf_str(str(v)))


def _align(n: int, a: int = GGUF_ALIGNMENT) -> int:
    return (n + a - 1) // a * a


# ── ftype policy ─────────────────────────────────────────────────────────────
def choose_type(name: str, shape, ftype: int) -> int:
    """Pick the on-disk ggml type for a tensor given the requested ftype."""
    if ftype == 0:
        return GGML_TYPE_F32
    if ftype == 2:
        return GGML_TYPE_BF16
    # ftype == 1: f16 for the big 2-D weight matrices, f32 for everything that
    # is precision-sensitive (norm gammas, modulation, biases, all 1-D).
    keep_f32 = ("gamma" in name) or ("modulation" in name) or ("norm" in name)
    if len(shape) >= 2 and not keep_f32:
        return GGML_TYPE_F16
    return GGML_TYPE_F32


def to_bytes(t, ggml_type: int) -> bytes:
    """torch tensor -> raw little-endian bytes in the chosen ggml type."""
    import torch
    t = t.detach().cpu().contiguous()
    if ggml_type == GGML_TYPE_F32:
        return t.float().numpy().astype("<f4", copy=False).tobytes()
    if ggml_type == GGML_TYPE_F16:
        return t.float().numpy().astype("<f2", copy=False).tobytes()
    if ggml_type == GGML_TYPE_BF16:
        # bf16 == upper 16 bits of f32, round-to-nearest-even.
        u32 = t.float().numpy().view(np.uint32)
        rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16
        return rounded.astype("<u2", copy=False).tobytes()
    raise ValueError(f"unhandled ggml type {ggml_type}")


def resolve_paths(args):
    model = args.model
    if model is None:
        hits = sorted(glob.glob(os.path.join(DEFAULT_SNAPSHOT, CKPT_STEM + ".safetensors")))
        if not hits:
            sys.exit("error: --model not given and no TRELLIS.2-4B snapshot found in HF cache")
        model = hits[-1]
    cfg = args.config or (os.path.splitext(model)[0] + ".json")
    out = args.output or (CKPT_STEM + ".gguf")
    return model, cfg, out


def main():
    ap = argparse.ArgumentParser(description="Convert TRELLIS.2 SS flow DiT to GGUF")
    ap.add_argument("--model", default=None, help="path to ...ss_flow...safetensors (default: HF cache)")
    ap.add_argument("--config", default=None, help="path to matching .json (default: alongside model)")
    ap.add_argument("--output", default=None, help="output .gguf (default: <stem>.gguf)")
    ap.add_argument("--pipeline-json", default=None, help="pipeline.json with shape_slat_normalization (default: alongside ckpts dir)")
    ap.add_argument("--ftype", type=int, default=1, choices=[0, 1, 2],
                    help="0=f32, 1=f16 (default), 2=bf16")
    args = ap.parse_args()

    from safetensors.torch import load_file

    model_path, cfg_path, out_path = resolve_paths(args)
    print(f"model : {model_path}")
    print(f"config: {cfg_path}")
    print(f"output: {out_path}  (ftype={args.ftype})")

    with open(cfg_path) as f:
        cfg = json.load(f)
    a = cfg["args"]
    print(f"arch  : {cfg.get('name')}  {a['num_blocks']} blocks, "
          f"d={a['model_channels']}, heads={a['num_heads']}, cond={a['cond_channels']}")

    rope_freq = a.get("rope_freq", (1.0, 10000.0))

    # ── KV metadata ──────────────────────────────────────────────────────────
    metadata = [
        kv_str("general.architecture", ARCH),
        kv_str("general.name", CKPT_STEM),
        kv_u32("general.file_type", args.ftype),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32(KV_PREFIX + "resolution",     a["resolution"]),
        kv_u32(KV_PREFIX + "in_channels",    a["in_channels"]),
        kv_u32(KV_PREFIX + "out_channels",   a["out_channels"]),
        kv_u32(KV_PREFIX + "model_channels", a["model_channels"]),
        kv_u32(KV_PREFIX + "cond_channels",  a["cond_channels"]),
        kv_u32(KV_PREFIX + "num_blocks",     a["num_blocks"]),
        kv_u32(KV_PREFIX + "num_heads",      a["num_heads"]),
        kv_f32(KV_PREFIX + "mlp_ratio",      a["mlp_ratio"]),
        kv_str(KV_PREFIX + "pe_mode",        a.get("pe_mode", "rope")),
        kv_bool(KV_PREFIX + "share_mod",            a.get("share_mod", False)),
        kv_bool(KV_PREFIX + "qk_rms_norm",          a.get("qk_rms_norm", False)),
        kv_bool(KV_PREFIX + "qk_rms_norm_cross",    a.get("qk_rms_norm_cross", False)),
        kv_f32(KV_PREFIX + "rope_freq_min",  float(rope_freq[0])),
        kv_f32(KV_PREFIX + "rope_freq_base", float(rope_freq[1])),
    ]

    # shape_slat_normalization (pipeline.json): the sampler output is
    # de-normalized with these before the decoder.
    pj_path = args.pipeline_json or os.path.join(
        os.path.dirname(os.path.dirname(model_path)), "pipeline.json")
    with open(pj_path) as f:
        norm = json.load(f)["args"]["shape_slat_normalization"]
    assert len(norm["mean"]) == a["out_channels"]
    for i, v in enumerate(norm["mean"]):
        metadata.append(kv_f32(KV_PREFIX + f"norm_mean.{i}", v))
    for i, v in enumerate(norm["std"]):
        metadata.append(kv_f32(KV_PREFIX + f"norm_std.{i}", v))

    # ── tensors ──────────────────────────────────────────────────────────────
    print("loading state_dict...")
    sd = load_file(model_path)

    tensors = []  # (name, ggml_type, dims_ggml_order, raw_bytes)
    counts = {GGML_TYPE_F32: 0, GGML_TYPE_F16: 0, GGML_TYPE_BF16: 0}
    for name in sorted(sd.keys()):
        t = sd[name]
        shape = tuple(t.shape)
        gtype = choose_type(name, shape, args.ftype)
        raw = to_bytes(t, gtype)
        dims = list(reversed(shape)) if len(shape) > 0 else [1]  # ggml ne[] order
        tensors.append((name, gtype, dims, raw))
        counts[gtype] += 1

    print(f"tensors: {len(tensors)}  "
          f"(f32={counts[GGML_TYPE_F32]}, f16={counts[GGML_TYPE_F16]}, bf16={counts[GGML_TYPE_BF16]})")

    # ── assemble header + infos, compute aligned data offsets ────────────────
    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for m in metadata:
        header += m

    infos = bytearray()
    offset = 0
    offsets = []
    for name, gtype, dims, raw in tensors:
        offsets.append(offset)
        offset = _align(offset + len(raw))
    for (name, gtype, dims, raw), off in zip(tensors, offsets):
        infos += _gguf_str(name)
        infos += struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", int(d))
        infos += struct.pack("<I", gtype)
        infos += struct.pack("<Q", off)

    pre_data = len(header) + len(infos)
    pad0 = _align(pre_data) - pre_data

    with open(out_path, "wb") as fout:
        fout.write(header)
        fout.write(infos)
        fout.write(b"\x00" * pad0)
        for (name, gtype, dims, raw), off in zip(tensors, offsets):
            cur = fout.tell()  # already aligned per loop invariant
            fout.write(raw)
            pad = _align(len(raw)) - len(raw)
            if pad:
                fout.write(b"\x00" * pad)

    print(f"wrote {out_path}  ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    main()
