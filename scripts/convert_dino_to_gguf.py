#!/usr/bin/env python3
"""
Convert the DINOv3 ViT-L/16 image-conditioning encoder (HF format,
facebook/dinov3-vitl16-pretrain-lvd1689m or a mirror) to a GGUF file for
trellis2.cpp.

TRELLIS.2 uses this model (transformers DINOv3ViTModel) to turn the
preprocessed 512x512 input image into the [1, 1029, 1024] conditioning tensor
consumed by the flow models: 1 CLS + 4 register + 32x32 patch tokens, last
transformer layer output passed through an affine-free layer norm (the model's
own final `norm` layer is NOT applied).

Architecture (config.json): hidden 1024, 24 layers, 16 heads, MLP 4096 (exact
GELU, not gated), patch 16, axial 2D RoPE over patch-center coords with
theta=100, LayerScale on both residual branches, q/v/o biases but no k bias.

Self-contained like the other converters (safetensors + numpy). Tensors keep
their HF state-dict names; hyperparameters travel as `trellis2.dino.*` KV.

Usage:
    python convert_dino_to_gguf.py --model models/dinov3-vitl16/model.safetensors \
        --output ggufs/dino_f32.gguf --ftype 0

ftype: 0 = f32 (validation), 1 = f16 (matrices f16; norms/biases/1-D f32).
"""

import argparse
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
GGUF_VT_FLOAT32 = 6
GGUF_VT_BOOL = 7
GGUF_VT_STRING = 8

ARCH = "trellis2-dino"
KV_PREFIX = "trellis2.dino."


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


def choose_type(name, shape, ftype: int) -> int:
    if ftype == 0:
        return GGML_TYPE_F32
    # matrices f16; 1-D (biases, norms, layerscale) and token embeddings f32
    if len(shape) < 2:
        return GGML_TYPE_F32
    if "cls_token" in name or "register_tokens" in name or "mask_token" in name:
        return GGML_TYPE_F32
    return GGML_TYPE_F16


def to_bytes(arr_f32: np.ndarray, ggml_type: int) -> bytes:
    if ggml_type == GGML_TYPE_F32:
        return arr_f32.astype("<f4", copy=False).tobytes()
    if ggml_type == GGML_TYPE_F16:
        return arr_f32.astype("<f2", copy=False).tobytes()
    raise ValueError(f"unhandled ggml type {ggml_type}")


def main():
    ap = argparse.ArgumentParser(description="Convert DINOv3 ViT-L/16 to GGUF")
    # models/ sits at the repository root, one level above scripts/ — resolving
    # this against __file__'s own directory looked for scripts/models/.
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "dinov3-vitl16", "model.safetensors"))
    ap.add_argument("--config", default=None, help="config.json (default: alongside model)")
    ap.add_argument("--output", default="dino.gguf")
    ap.add_argument("--ftype", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    from safetensors.numpy import load_file

    cfg_path = args.config or os.path.join(os.path.dirname(args.model), "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    if cfg.get("model_type") != "dinov3_vit":
        sys.exit(f"error: unexpected model_type {cfg.get('model_type')!r}")
    if cfg.get("use_gated_mlp"):
        sys.exit("error: gated MLP variant not supported")

    print(f"model : {args.model}")
    print(f"output: {args.output}  (ftype={args.ftype})")
    print(f"arch  : hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']} "
          f"heads={cfg['num_attention_heads']} mlp={cfg['intermediate_size']} "
          f"patch={cfg['patch_size']} registers={cfg['num_register_tokens']} "
          f"rope_theta={cfg['rope_theta']}")

    metadata = [
        kv_str("general.architecture", ARCH),
        kv_str("general.name", "dinov3-vitl16-pretrain-lvd1689m"),
        kv_u32("general.file_type", args.ftype),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32(KV_PREFIX + "hidden_size",         cfg["hidden_size"]),
        kv_u32(KV_PREFIX + "n_layers",            cfg["num_hidden_layers"]),
        kv_u32(KV_PREFIX + "n_heads",             cfg["num_attention_heads"]),
        kv_u32(KV_PREFIX + "intermediate_size",   cfg["intermediate_size"]),
        kv_u32(KV_PREFIX + "patch_size",          cfg["patch_size"]),
        kv_u32(KV_PREFIX + "num_register_tokens", cfg["num_register_tokens"]),
        kv_f32(KV_PREFIX + "layer_norm_eps",      cfg["layer_norm_eps"]),
        kv_f32(KV_PREFIX + "rope_theta",          cfg["rope_theta"]),
        kv_bool(KV_PREFIX + "key_bias",           cfg.get("key_bias", False)),
        # ImageNet normalization baked in so inference needs no external config.
        kv_f32(KV_PREFIX + "image_mean.0", 0.485),
        kv_f32(KV_PREFIX + "image_mean.1", 0.456),
        kv_f32(KV_PREFIX + "image_mean.2", 0.406),
        kv_f32(KV_PREFIX + "image_std.0",  0.229),
        kv_f32(KV_PREFIX + "image_std.1",  0.224),
        kv_f32(KV_PREFIX + "image_std.2",  0.225),
    ]

    print("loading state_dict...")
    sd = load_file(args.model)

    tensors = []
    counts = {GGML_TYPE_F32: 0, GGML_TYPE_F16: 0}
    for name in sorted(sd.keys()):
        arr = np.asarray(sd[name]).astype(np.float32)
        # squeeze leading singleton batch dims of the token embeddings so they
        # arrive as [n_tokens, hidden] / [hidden]
        if name.endswith(("cls_token", "mask_token", "register_tokens")):
            arr = arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.ndim == 3 else arr
        shape = tuple(arr.shape)
        # patch_embeddings.weight [OC, IC, KH, KW]: bytes already match ggml's
        # ne=[KW, KH, IC, OC] expectation — dims just get written reversed.
        gtype = choose_type(name, shape, args.ftype)
        raw = to_bytes(np.ascontiguousarray(arr), gtype)
        dims = list(reversed(shape)) if len(shape) > 0 else [1]
        tensors.append((name, gtype, dims, raw))
        counts[gtype] += 1

    print(f"tensors: {len(tensors)}  (f32={counts[GGML_TYPE_F32]}, f16={counts[GGML_TYPE_F16]})")

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

    with open(args.output, "wb") as fout:
        fout.write(header)
        fout.write(infos)
        fout.write(b"\x00" * pad0)
        for (name, gtype, dims, raw), off in zip(tensors, offsets):
            fout.write(raw)
            pad = _align(len(raw)) - len(raw)
            if pad:
                fout.write(b"\x00" * pad)

    print(f"wrote {args.output}  ({os.path.getsize(args.output):,} bytes)")


if __name__ == "__main__":
    main()
