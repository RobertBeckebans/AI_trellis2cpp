#!/usr/bin/env python3
"""
Convert a TRELLIS.2 texture-SLAT flow DiT checkpoint
(slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16.safetensors) to GGUF.

Architecturally identical to the shape SLAT flow (same 1536-d / 30-block /
12-head sparse DiT, adaLN-Zero share_mod, self+cross attention to DINOv3 cond,
3D RoPE, QK-RMSNorm) with ONE difference: it is conditioned on the shape via
`concat_cond`. So in_channels = 64 = 32 tex-noise + 32 shape-SLAT, out = 32.
At sample time the (normalized) shape SLAT is concatenated onto the noise before
the input layer each step.

Emits the same `trellis2.slat_flow.*` KV as the shape-flow converter so the C++
loader is shared, plus:
  * trellis2.slat_flow.concat_cond_channels = 32   (0 on the shape flow)
  * norm_mean/std          = tex_slat_normalization   (output de-norm, 32)
  * concat_norm_mean/std   = shape_slat_normalization  (concat_cond in-norm, 32)
both read from texturing_pipeline.json.

Usage:
    python convert_tex_flow_to_gguf.py \
        --model models/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors \
        --texturing-json models/TRELLIS.2-4B/texturing_pipeline.json \
        --output ggufs/tex_slat_flow_512_f16.gguf --ftype 1
"""

import argparse
import json
import os
import struct

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
GGUF_VT_UINT32 = 4
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL = 7
GGUF_VT_STRING = 8

ARCH = "trellis2-slat-flow"          # shared loader with the shape SLAT flow
KV_PREFIX = "trellis2.slat_flow."


def _gguf_str(s): b = s.encode("utf-8"); return struct.pack("<Q", len(b)) + b
def _kv(key, vtype, payload): return _gguf_str(key) + struct.pack("<I", vtype) + payload
def kv_u32(key, v):  return _kv(key, GGUF_VT_UINT32,  struct.pack("<I", int(v)))
def kv_f32(key, v):  return _kv(key, GGUF_VT_FLOAT32, struct.pack("<f", float(v)))
def kv_bool(key, v): return _kv(key, GGUF_VT_BOOL,    struct.pack("<?", bool(v)))
def kv_str(key, v):  return _kv(key, GGUF_VT_STRING,  _gguf_str(str(v)))
def _align(n, a=GGUF_ALIGNMENT): return (n + a - 1) // a * a


def choose_type(name, shape, ftype):
    if ftype == 0: return GGML_TYPE_F32
    if ftype == 2: return GGML_TYPE_BF16
    keep_f32 = ("gamma" in name) or ("modulation" in name) or ("norm" in name)
    return GGML_TYPE_F16 if (len(shape) >= 2 and not keep_f32) else GGML_TYPE_F32


def to_bytes(t, ggml_type):
    t = t.detach().cpu().contiguous()
    if ggml_type == GGML_TYPE_F32: return t.float().numpy().astype("<f4", copy=False).tobytes()
    if ggml_type == GGML_TYPE_F16: return t.float().numpy().astype("<f2", copy=False).tobytes()
    if ggml_type == GGML_TYPE_BF16:
        u32 = t.float().numpy().view(np.uint32)
        return (((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype("<u2", copy=False)).tobytes()
    raise ValueError(ggml_type)


def main():
    ap = argparse.ArgumentParser(description="Convert TRELLIS.2 texture SLAT flow DiT to GGUF")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--texturing-json", default=None,
                    help="texturing_pipeline.json (tex + shape slat normalization)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ftype", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()

    from safetensors.torch import load_file

    cfg_path = args.config or (os.path.splitext(args.model)[0] + ".json")
    with open(cfg_path) as f:
        a = json.load(f)["args"]
    print(f"model : {args.model}")
    print(f"output: {args.output}  (ftype={args.ftype})")
    print(f"arch  : in={a['in_channels']} out={a['out_channels']} cond={a['cond_channels']} "
          f"d={a['model_channels']} blks={a['num_blocks']} heads={a['num_heads']}")

    concat_cond_channels = a["in_channels"] - a["out_channels"]
    assert concat_cond_channels > 0, "tex flow must have concat_cond (in > out)"

    tj = args.texturing_json or os.path.join(os.path.dirname(os.path.dirname(args.model)),
                                             "texturing_pipeline.json")
    with open(tj) as f:
        tpa = json.load(f)["args"]
    tex_norm = tpa["tex_slat_normalization"]
    shape_norm = tpa["shape_slat_normalization"]
    assert len(tex_norm["mean"]) == a["out_channels"], "tex_slat_norm must match out_channels"
    assert len(shape_norm["mean"]) == concat_cond_channels, "shape_slat_norm must match concat_cond"

    rope_freq = a.get("rope_freq", (1.0, 10000.0))
    metadata = [
        kv_str("general.architecture", ARCH),
        kv_str("general.name", os.path.splitext(os.path.basename(args.model))[0]),
        kv_u32("general.file_type", args.ftype),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32(KV_PREFIX + "resolution",     a["resolution"]),
        kv_u32(KV_PREFIX + "in_channels",    a["in_channels"]),
        kv_u32(KV_PREFIX + "out_channels",   a["out_channels"]),
        kv_u32(KV_PREFIX + "concat_cond_channels", concat_cond_channels),
        kv_u32(KV_PREFIX + "model_channels", a["model_channels"]),
        kv_u32(KV_PREFIX + "cond_channels",  a["cond_channels"]),
        kv_u32(KV_PREFIX + "num_blocks",     a["num_blocks"]),
        kv_u32(KV_PREFIX + "num_heads",      a["num_heads"]),
        kv_f32(KV_PREFIX + "mlp_ratio",      a["mlp_ratio"]),
        kv_str(KV_PREFIX + "pe_mode",        a.get("pe_mode", "rope")),
        kv_bool(KV_PREFIX + "share_mod",         a.get("share_mod", False)),
        kv_bool(KV_PREFIX + "qk_rms_norm",       a.get("qk_rms_norm", False)),
        kv_bool(KV_PREFIX + "qk_rms_norm_cross", a.get("qk_rms_norm_cross", False)),
        kv_f32(KV_PREFIX + "rope_freq_min",  float(rope_freq[0])),
        kv_f32(KV_PREFIX + "rope_freq_base", float(rope_freq[1])),
    ]
    # output de-normalization (tex slat) and concat_cond in-normalization (shape slat)
    for i, v in enumerate(tex_norm["mean"]):   metadata.append(kv_f32(KV_PREFIX + f"norm_mean.{i}", v))
    for i, v in enumerate(tex_norm["std"]):    metadata.append(kv_f32(KV_PREFIX + f"norm_std.{i}", v))
    for i, v in enumerate(shape_norm["mean"]): metadata.append(kv_f32(KV_PREFIX + f"concat_norm_mean.{i}", v))
    for i, v in enumerate(shape_norm["std"]):  metadata.append(kv_f32(KV_PREFIX + f"concat_norm_std.{i}", v))

    print("loading state_dict...")
    sd = load_file(args.model)
    tensors, counts = [], {GGML_TYPE_F32: 0, GGML_TYPE_F16: 0, GGML_TYPE_BF16: 0}
    for name in sorted(sd.keys()):
        t = sd[name]; shape = tuple(t.shape)
        gtype = choose_type(name, shape, args.ftype)
        raw = to_bytes(t, gtype)
        dims = list(reversed(shape)) if len(shape) > 0 else [1]
        tensors.append((name, gtype, dims, raw)); counts[gtype] += 1
    print(f"tensors: {len(tensors)}  (f32={counts[GGML_TYPE_F32]}, f16={counts[GGML_TYPE_F16]}, bf16={counts[GGML_TYPE_BF16]})")

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata))
    for m in metadata: header += m

    infos = bytearray()
    offset, offsets = 0, []
    for name, gtype, dims, raw in tensors:
        offsets.append(offset); offset = _align(offset + len(raw))
    for (name, gtype, dims, raw), off in zip(tensors, offsets):
        infos += _gguf_str(name); infos += struct.pack("<I", len(dims))
        for d in dims: infos += struct.pack("<Q", int(d))
        infos += struct.pack("<I", gtype); infos += struct.pack("<Q", off)

    pre_data = len(header) + len(infos)
    pad0 = _align(pre_data) - pre_data
    with open(args.output, "wb") as fout:
        fout.write(header); fout.write(infos); fout.write(b"\x00" * pad0)
        for (name, gtype, dims, raw), off in zip(tensors, offsets):
            fout.write(raw)
            pad = _align(len(raw)) - len(raw)
            if pad: fout.write(b"\x00" * pad)
    print(f"wrote {args.output}  ({os.path.getsize(args.output):,} bytes)")


if __name__ == "__main__":
    main()
