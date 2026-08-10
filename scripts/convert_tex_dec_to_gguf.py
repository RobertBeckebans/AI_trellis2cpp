#!/usr/bin/env python3
"""
Convert the TRELLIS.2 texture-SLAT VAE decoder checkpoint
(tex_dec_next_dc_f16c32_fp16.safetensors) to a GGUF file for trellis2.cpp.

This is a SparseUnetVaeDecoder — the SAME sparse ConvNeXt U-Net as the shape
decoder (FlexiDualGridVaeDecoder) except:
  * out_channels = 6  (PBR: base_color[3], metallic[1], roughness[1], alpha[1]),
    no dual-grid geometry head;
  * pred_subdiv = False  -> NO `to_subdiv` layers. The decoder does not predict
    which children to keep; it replays the subdivision structure recorded by the
    shape encoder's Spatial2Channel steps (threaded through the SparseTensor
    spatial cache). So the tex decoder reconstructs exactly the encoder's
    res-1024 voxel set; the C++ side must supply that per-level subdivision.

Same architecture/weight layout as the shape decoder otherwise:
    model_channels [1024,512,256,128,64], num_blocks [4,16,8,4,0],
    SparseConvNeXtBlock3d blocks, SparseResBlockC2S3d up-blocks.
Conv weights are FlexGEMM [Co,kD,kH,kW,Ci] -> reshaped to [Co, kD*kH*kW, Ci].

Usage: python convert_tex_dec_to_gguf.py --output ggufs/tex_dec_f16.gguf --ftype 1
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
GGUF_VT_UINT32 = 4
GGUF_VT_FLOAT32 = 6
GGUF_VT_STRING = 8

ARCH = "trellis2-tex-dec"
KV_PREFIX = "trellis2.tex_dec."


def _gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key, vtype, payload):
    return _gguf_str(key) + struct.pack("<I", vtype) + payload


def kv_u32(key, v):  return _kv(key, GGUF_VT_UINT32,  struct.pack("<I", int(v)))
def kv_f32(key, v):  return _kv(key, GGUF_VT_FLOAT32, struct.pack("<f", float(v)))
def kv_str(key, v):  return _kv(key, GGUF_VT_STRING,  _gguf_str(str(v)))


def _align(n, a=GGUF_ALIGNMENT):
    return (n + a - 1) // a * a


def main():
    ap = argparse.ArgumentParser(description="Convert TRELLIS.2 texture decoder to GGUF")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__),
                    "models", "TRELLIS.2-4B", "ckpts", "tex_dec_next_dc_f16c32_fp16.safetensors"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default="tex_dec.gguf")
    ap.add_argument("--ftype", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    from safetensors.torch import load_file

    cfg_path = args.config or (os.path.splitext(args.model)[0] + ".json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    a = cfg["args"]
    channels = a["model_channels"]
    nblocks = a["num_blocks"]
    out_channels = a["out_channels"]
    assert out_channels == 6, f"expected 6 PBR channels, got {out_channels}"
    assert a.get("pred_subdiv", True) is False, "tex decoder must be pred_subdiv=False"
    print(f"model : {args.model}")
    print(f"output: {args.output}  (ftype={args.ftype})")
    print(f"arch  : channels={channels} blocks={nblocks} latent={a['latent_channels']} out={out_channels}")

    metadata = [
        kv_str("general.architecture", ARCH),
        kv_str("general.name", "tex_dec_next_dc_f16c32_fp16"),
        kv_u32("general.file_type", args.ftype),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_u32(KV_PREFIX + "latent_channels", a["latent_channels"]),
        kv_u32(KV_PREFIX + "out_channels",    out_channels),
        kv_u32(KV_PREFIX + "n_levels",        len(channels)),
        kv_f32(KV_PREFIX + "norm_eps",        1e-6),
    ]
    for i, ch in enumerate(channels):
        metadata.append(kv_u32(KV_PREFIX + f"channels.{i}", ch))
    for i, nb in enumerate(nblocks):
        metadata.append(kv_u32(KV_PREFIX + f"num_blocks.{i}", nb))

    print("loading state_dict...")
    sd = load_file(args.model)

    tensors = []
    counts = {GGML_TYPE_F32: 0, GGML_TYPE_F16: 0}
    for name in sorted(sd.keys()):
        arr = sd[name].float().numpy().astype(np.float32)
        shape = tuple(arr.shape)
        if arr.ndim == 5:  # FlexGEMM conv [Co,kD,kH,kW,Ci] -> [Co, 27, Ci]
            Co, kD, kH, kW, Ci = shape
            arr = np.ascontiguousarray(arr).reshape(Co, kD * kH * kW, Ci)
            shape = arr.shape
        gtype = GGML_TYPE_F32
        if args.ftype == 1 and len(shape) >= 2:
            gtype = GGML_TYPE_F16
        raw = (arr.astype("<f2") if gtype == GGML_TYPE_F16 else arr.astype("<f4")).tobytes()
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
