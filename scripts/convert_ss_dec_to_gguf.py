#!/usr/bin/env python3
"""
Convert the TRELLIS.2 sparse-structure DECODER checkpoint
(ss_dec_conv3d_16l8_fp16.safetensors) to a GGUF file for trellis2.cpp.

This is the stage-1 decoder D_S: a dense 3D-conv ResNet that turns the
sparse-structure latent z_s ([8, 16, 16, 16]) into an occupancy logit grid at
64^3 ([1, 64, 64, 64]). It upsamples 16 -> 32 -> 64 with two pixel-shuffle
blocks. Architecture (config ss_dec_conv3d_16l8_fp16.json):

    SparseStructureDecoder(out_channels=1, latent_channels=8,
                           num_res_blocks=2, num_res_blocks_middle=2,
                           channels=[512, 128, 32], norm_type="layer")

Like convert_ss_flow_to_gguf.py this is self-contained (safetensors + numpy +
torch) and writes a standard GGUF v3 file read back by ggml's
gguf_init_from_file(). Hyperparameters travel as KV metadata under the
`trellis2.ss_dec.*` namespace; tensors keep their original checkpoint names.

Conv3d weights are 5-D in PyTorch ([OC, IC, kD, kH, kW]). ggml tensors are 4-D,
and ggml_conv_3d_direct wants the kernel as ne = [kW, kH, kD, IC*OC] with the
merged channel index packed oc*IC + ic. A C-contiguous [OC, IC, kD, kH, kW]
array reshaped to [OC*IC, kD, kH, kW] is exactly that packing (and identical
bytes), so we just reshape before writing — no permute, no data movement.

Usage:
    python convert_ss_dec_to_gguf.py --output ss_dec.gguf --ftype 0
    # --model/--config default to the microsoft/TRELLIS-image-large HF snapshot.

ftype: 0 = f32 (lossless upcast from the fp16 checkpoint; use for validation),
       1 = f16 (default; conv weight matrices f16, norms/biases f32).
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

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGUF_VT_UINT32 = 4
GGUF_VT_INT32 = 5
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL = 7
GGUF_VT_STRING = 8

ARCH = "trellis2-ss-dec"
KV_PREFIX = "trellis2.ss_dec."

DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/*/ckpts"
)
CKPT_STEM = "ss_dec_conv3d_16l8_fp16"


# ── GGUF writer (minimal, v3) ────────────────────────────────────────────────
def _gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _gguf_str(key) + struct.pack("<I", vtype) + payload


def kv_u32(key, v):  return _kv(key, GGUF_VT_UINT32,  struct.pack("<I", int(v)))
def kv_f32(key, v):  return _kv(key, GGUF_VT_FLOAT32, struct.pack("<f", float(v)))
def kv_bool(key, v): return _kv(key, GGUF_VT_BOOL,    struct.pack("<?", bool(v)))
def kv_str(key, v):  return _kv(key, GGUF_VT_STRING,  _gguf_str(str(v)))


def _align(n: int, a: int = GGUF_ALIGNMENT) -> int:
    return (n + a - 1) // a * a


def choose_type(shape, ftype: int) -> int:
    """f32 always for ftype 0; for ftype 1 the big conv weight matrices are f16
    while everything 1-D (biases, norm weight/bias) stays f32."""
    if ftype == 0:
        return GGML_TYPE_F32
    return GGML_TYPE_F16 if len(shape) >= 2 else GGML_TYPE_F32


def to_bytes(arr_f32: np.ndarray, ggml_type: int) -> bytes:
    if ggml_type == GGML_TYPE_F32:
        return arr_f32.astype("<f4", copy=False).tobytes()
    if ggml_type == GGML_TYPE_F16:
        return arr_f32.astype("<f2", copy=False).tobytes()
    raise ValueError(f"unhandled ggml type {ggml_type}")


def resolve_paths(args):
    model = args.model
    if model is None:
        hits = sorted(glob.glob(os.path.join(DEFAULT_SNAPSHOT, CKPT_STEM + ".safetensors")))
        if not hits:
            sys.exit("error: --model not given and no TRELLIS-image-large snapshot in HF cache")
        model = hits[-1]
    cfg = args.config or (os.path.splitext(model)[0] + ".json")
    out = args.output or (CKPT_STEM + ".gguf")
    return model, cfg, out


def main():
    ap = argparse.ArgumentParser(description="Convert TRELLIS.2 SS decoder to GGUF")
    ap.add_argument("--model", default=None, help="path to ...ss_dec...safetensors (default: HF cache)")
    ap.add_argument("--config", default=None, help="path to matching .json (default: alongside model)")
    ap.add_argument("--output", default=None, help="output .gguf (default: <stem>.gguf)")
    ap.add_argument("--ftype", type=int, default=1, choices=[0, 1],
                    help="0=f32 (lossless, for validation), 1=f16 (default)")
    args = ap.parse_args()

    from safetensors.numpy import load_file

    model_path, cfg_path, out_path = resolve_paths(args)
    print(f"model : {model_path}")
    print(f"config: {cfg_path}")
    print(f"output: {out_path}  (ftype={args.ftype})")

    with open(cfg_path) as f:
        cfg = json.load(f)
    a = cfg["args"]
    channels = a["channels"]
    print(f"arch  : {cfg.get('name')}  channels={channels}, "
          f"res_blocks={a['num_res_blocks']}, mid={a['num_res_blocks_middle']}, "
          f"latent={a['latent_channels']}, out={a['out_channels']}")

    # ── KV metadata ──────────────────────────────────────────────────────────
    metadata = [
        kv_str("general.architecture", ARCH),
        kv_str("general.name", CKPT_STEM),
        kv_u32("general.file_type", args.ftype),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32(KV_PREFIX + "out_channels",          a["out_channels"]),
        kv_u32(KV_PREFIX + "latent_channels",       a["latent_channels"]),
        kv_u32(KV_PREFIX + "num_res_blocks",        a["num_res_blocks"]),
        kv_u32(KV_PREFIX + "num_res_blocks_middle", a["num_res_blocks_middle"]),
        kv_u32(KV_PREFIX + "n_levels",              len(channels)),
        kv_str(KV_PREFIX + "norm_type",             a.get("norm_type", "layer")),
        kv_f32(KV_PREFIX + "norm_eps",              1e-5),
    ]
    for i, ch in enumerate(channels):
        metadata.append(kv_u32(KV_PREFIX + f"channels.{i}", ch))

    # ── tensors ──────────────────────────────────────────────────────────────
    print("loading state_dict...")
    sd = load_file(model_path)   # numpy arrays (fp16/fp32 as stored)

    tensors = []  # (name, ggml_type, dims_ggml_order, raw_bytes)
    counts = {GGML_TYPE_F32: 0, GGML_TYPE_F16: 0}
    for name in sorted(sd.keys()):
        arr = np.asarray(sd[name]).astype(np.float32)
        shape = tuple(arr.shape)
        if arr.ndim == 5:  # Conv3d weight [OC, IC, kD, kH, kW] -> [OC*IC, kD, kH, kW]
            OC, IC, kD, kH, kW = shape
            arr = np.ascontiguousarray(arr).reshape(OC * IC, kD, kH, kW)
            shape = arr.shape
        gtype = choose_type(shape, args.ftype)
        raw = to_bytes(np.ascontiguousarray(arr), gtype)
        dims = list(reversed(shape)) if len(shape) > 0 else [1]  # ggml ne[] order
        tensors.append((name, gtype, dims, raw))
        counts[gtype] += 1

    print(f"tensors: {len(tensors)}  (f32={counts[GGML_TYPE_F32]}, f16={counts[GGML_TYPE_F16]})")

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
            fout.write(raw)
            pad = _align(len(raw)) - len(raw)
            if pad:
                fout.write(b"\x00" * pad)

    print(f"wrote {out_path}  ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    main()
