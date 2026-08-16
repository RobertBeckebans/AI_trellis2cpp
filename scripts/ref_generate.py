#!/usr/bin/env python3
"""Run the PyTorch reference geometry pipeline end to end and dump its voxels.

This is `backend-parity` phase 1c: a mesh produced by the reference
implementation, so it can sit under `generations/` next to the ROCm, Vulkan and
CPU runs of the same image and be judged in the same viewer.

The stages below mirror `Trellis2ImageTo3DPipeline.run()` from the upstream
package (see the `trellis2/` checkout in the repo root) for pipeline types
"512" and "1024_cascade", but stop one step earlier: `run()` ends by handing the
decoder's 7-channel output to o_voxel's CUDA mesher, which does not exist on
Windows. We dump those 7 channels instead and mesh them with our own extractor
(examples/dual_grid_cli.cpp, the same code every backend uses), so the only
thing that differs between this artefact and a trellis2.cpp run is the network
— not the mesher.

Faithfulness notes:
  * The models run in true fp32 (ref_common.setup disables TF32 and the reduced
    precision SDPA kernels), as every other reference dump in this repo does.
  * The RNG draw order matches `run()`: one `torch.manual_seed(seed)`, then the
    sparse-structure noise, then the shape-SLAT noise, all drawn on CPU. So the
    same seed means the same noise the reference implementation would use.
  * Preprocessing is `pipeline.preprocess_image` for an image that carries
    alpha; an opaque image would need the rembg model and is rejected here.

A checkpoint is written once the SLAT is sampled, before the decode — the
decode is the step that can run the host out of memory, and it is preceded by
GPU work measured in minutes. `--resume` picks that up and goes straight to the
decode.

Usage:
    uv run --extra rocm python scripts/ref_generate.py \
        --image assets/einstein.png --seed 42 --pipeline-type 512
    uv run --extra rocm python scripts/ref_generate.py ... --resume
"""

import argparse
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_common  # noqa: E402

ref_common.setup()  # TF32 off + sdpa sparse attention + pure-torch sparse conv

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

# The dual-grid dump the C++ mesher reads. Deliberately minimal: it exists only
# to hand one decoder output to one extractor.
#   magic[8] "FDGVOX01" | u32 n | u32 grid_size | f32 margin
#   i32[3n] coords (x,y,z) | f32[7n] feats (voxel-major, raw decoder channels)
FDGVOX_MAGIC = b"FDGVOX01"


def write_fdgvox(path, coords, feats, grid_size, margin):
    coords = np.ascontiguousarray(coords, dtype="<i4")
    feats = np.ascontiguousarray(feats, dtype="<f4")
    assert coords.shape[0] == feats.shape[0], (coords.shape, feats.shape)
    assert coords.shape[1] == 3 and feats.shape[1] == 7, (coords.shape, feats.shape)
    with open(path, "wb") as f:
        f.write(FDGVOX_MAGIC)
        f.write(struct.pack("<IIf", coords.shape[0], grid_size, margin))
        f.write(coords.tobytes())
        f.write(feats.tobytes())


def load_flow(stem, dev):
    """SLatFlowModel in fp32 (mirrors dump_cascade_reference.load_flow)."""
    from safetensors.torch import load_file
    from trellis2.models.structured_latent_flow import SLatFlowModel
    with open(stem + ".json") as f:
        cfg = json.load(f)["args"]
    cfg.pop("initialization", None)
    cfg.pop("dtype", None)
    m = SLatFlowModel(**cfg, dtype="float32")
    sd = {k: v.float() for k, v in load_file(stem + ".safetensors").items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    m.convert_to(torch.float32)
    m.eval().to(dev)
    return m, cfg


def decode_and_write(dec, slat, resolution, margin, dec_dev, scaffold, args, timings, stage, t0,
                     ss_params, slat_params):
    """The base SparseUnetVaeDecoder forward plus the dumps.

    Run directly rather than through `dec(...)`, which would go on to call
    o_voxel's CUDA mesher. Shared by the fresh and the resumed path so the two
    cannot drift apart.
    """
    with torch.no_grad():
        h = dec.from_latent(slat.to(dec_dev).float())
        for i, res in enumerate(dec.blocks):
            for j, block in enumerate(res):
                if i < len(dec.blocks) - 1 and j == len(res) - 1:
                    h, _sub = block(h)
                else:
                    h = block(h)
            print(f"decode level {i}: {h.feats.shape[0]} voxels x {h.feats.shape[1]} ch")
        hn = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        out7 = dec.output_layer(hn)
    t0 = stage(f"decoding shape ({resolution})", t0)

    feats = out7.feats.detach().cpu().numpy()
    vox = out7.coords.detach().cpu().numpy()[:, 1:]
    print(f"out7: {feats.shape}, intersected frac="
          f"{(feats[:, 3:6] > 0).mean():.4f}, grid {resolution}^3, margin {margin}")
    write_fdgvox(args.out, vox, feats, resolution, margin)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "image": os.path.abspath(args.image),
                "seed": args.seed,
                "pipelineType": args.pipeline_type,
                "resolution": resolution,
                "voxels": int(feats.shape[0]),
                "scaffold": int(scaffold),
                "device": args.device,
                "decoderDevice": args.decoder_device,
                "stageTimings": timings,
                "ssSamplerParams": ss_params,
                "slatSamplerParams": slat_params,
            }, f, indent=2)


def load_decoder(models_dir, resolution, dec_dev):
    """FlexiDualGridVaeDecoder in fp32, on the device that has to hold the
    expansion. The cascade needs it before the HR sampler (for `upsample`), and
    a resumed run needs it without having sampled anything, so it is built
    separately from either."""
    from safetensors.torch import load_file
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    stem = os.path.join(models_dir, "shape_dec_next_dc_f16c32_fp16")
    with open(stem + ".json") as f:
        cfg = json.load(f)["args"]
    cfg.pop("use_fp16", None)
    cfg.pop("resolution", None)
    dec = FlexiDualGridVaeDecoder(resolution=resolution, use_fp16=False, **cfg)
    dec.load_state_dict({k: v.float() for k, v in load_file(stem + ".safetensors").items()})
    dec.eval().float().to(dec_dev)
    return dec


def save_slat_state(path, slat, resolution, scaffold, pipeline_type):
    """Checkpoint taken once the SLAT exists, before the decode.

    The decode is the one step that can run the host out of memory, and it is
    preceded by ~15 minutes of GPU work (two or three samplers plus, in the
    cascade, the scaffold upsample). Losing that to an OOM is avoidable, so it
    is not left to chance.
    """
    np.savez(path,
             feats=slat.feats.detach().cpu().numpy().astype(np.float32),
             coords=slat.coords.detach().cpu().numpy().astype(np.int32),
             resolution=np.int64(resolution),
             scaffold=np.int64(scaffold),
             pipeline_type=np.str_(pipeline_type))
    print(f"checkpoint: {path} ({os.path.getsize(path):,} bytes)")


def load_slat_state(path, sp, dev):
    z = np.load(path, allow_pickle=False)
    slat = sp.SparseTensor(feats=torch.from_numpy(z["feats"]).to(dev),
                           coords=torch.from_numpy(z["coords"]).to(dev))
    return slat, int(z["resolution"]), int(z["scaffold"])


def encode_image(model, pre, resolution, dev):
    """DinoV3FeatureExtractor.__call__ for a single PIL image, spelled out.

    Same path as scripts/dump_dino_reference.py, whose output the C++ DINO stage
    is already validated against — so the conditioning here is the one the
    parity tests use, not a second implementation of it.
    """
    resized = pre.resize((resolution, resolution), Image.Resampling.LANCZOS)
    x = np.array(resized.convert("RGB")).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixel_values = ((x - mean) / std).to(dev)
    with torch.no_grad():
        hidden = model.embeddings(pixel_values, bool_masked_pos=None)
        rope = model.rope_embeddings(pixel_values)
        for layer in model.layer:
            hidden = layer(hidden, position_embeddings=rope)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        cond = F.layer_norm(hidden, hidden.shape[-1:])
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGBA input image (alpha is the matte)")
    ap.add_argument("--models", default=os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "ckpts"))
    ap.add_argument("--dino", default=os.path.join(ref_common.MODELS, "dinov3-vitl16"))
    ap.add_argument("--ss-dec", default=os.path.join(ref_common.MODELS, "TRELLIS-image-large",
                                                    "ckpts", "ss_dec_conv3d_16l8_fp16"))
    ap.add_argument("--pipeline-json", default=os.path.join(ref_common.MODELS, "TRELLIS.2-4B",
                                                           "pipeline.json"))
    ap.add_argument("--pipeline-type", default="512", choices=["512", "1024_cascade"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--decoder-device", default="cpu",
                    help="the shape decoder expands to millions of voxels through the "
                         "pure-torch sparse conv; cpu is what fits and what the C++ "
                         "port also pins this stage to")
    ap.add_argument("--max-num-tokens", type=int, default=49152)
    ap.add_argument("--out", default=os.path.join(ref_common.DUMPS, "ref_generate.fdgvox"))
    ap.add_argument("--out-preprocessed", default=None,
                    help="write the preprocessed RGB image here (for the generation record)")
    ap.add_argument("--out-json", default=None,
                    help="write stage timings and counts here (for the generation record)")
    ap.add_argument("--state", default=None,
                    help="checkpoint written once the SLAT is sampled, before the decode "
                         "(default: <out>.state.npz). The decode is the step that can run "
                         "the host out of memory, and everything before it is ~15 minutes "
                         "of GPU work that should not have to be repeated.")
    ap.add_argument("--resume", action="store_true",
                    help="skip straight to the decode using --state, if it exists")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    state_path = args.state or (os.path.splitext(args.out)[0] + ".state.npz")

    from safetensors.torch import load_file
    from transformers import DINOv3ViTModel
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2.modules import sparse as sp

    dev = torch.device(args.device)
    dec_dev = torch.device(args.decoder_device)
    timings = []

    def stage(name, t0):
        ms = int((time.time() - t0) * 1000)
        timings.append({"stage": name, "milliseconds": ms})
        print(f"[{name}] {ms/1000:.1f}s")
        return time.time()

    with open(args.pipeline_json) as f:
        pj = json.load(f)["args"]
    ss_params = pj["sparse_structure_sampler"]["params"]
    slat_params = pj["shape_slat_sampler"]["params"]
    norm = pj["shape_slat_normalization"]
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)

    # ── resume: everything before the decode is already on disk ──────────────
    if args.resume and os.path.isfile(state_path):
        t0 = time.time()
        slat, resolution, scaffold = load_slat_state(state_path, sp, dec_dev)
        dec = load_decoder(args.models, resolution, dec_dev)
        print(f"resumed from {state_path}: {slat.feats.shape[0]} tokens, "
              f"resolution {resolution}, scaffold {scaffold}")
        t0 = stage("resume from checkpoint", t0)
        decode_and_write(dec, slat, resolution, float(dec.voxel_margin), dec_dev, scaffold,
                         args, timings, stage, t0, ss_params, slat_params)
        return
    if args.resume:
        print(f"--resume given but {state_path} does not exist; running from the start")

    # ── preprocess (pipeline.preprocess_image, alpha branch) ─────────────────
    t0 = time.time()
    img = Image.open(args.image).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    if np.all(alpha == 255):
        raise SystemExit("input is fully opaque; the reference would run rembg here, "
                         "which this driver does not load — supply a cut-out RGBA image")
    pre = ref_common.preprocess_rgba(img)
    if args.out_preprocessed:
        pre.save(args.out_preprocessed)
    t0 = stage("preprocess", t0)

    # ── conditioning ─────────────────────────────────────────────────────────
    dino = DINOv3ViTModel.from_pretrained(args.dino)
    dino.eval().float().to(dev)
    cond_512 = encode_image(dino, pre, 512, dev)
    cond_1024 = encode_image(dino, pre, 1024, dev) if args.pipeline_type != "512" else None
    del dino
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"cond_512: {tuple(cond_512.shape)}")
    t0 = stage("encoding image (DINOv3)", t0)

    # run() seeds once, here, and every noise below is drawn on CPU in this
    # order — so this reproduces the reference implementation's noise exactly.
    torch.manual_seed(args.seed)

    # ── sparse structure: flow sampler + decoder → 32^3 scaffold ─────────────
    ss_stem = os.path.join(args.models, "ss_flow_img_dit_1_3B_64_bf16")
    with open(ss_stem + ".json") as f:
        ss_cfg = json.load(f)["args"]
    ss_flow = SparseStructureFlowModel(**ss_cfg)
    ss_flow.convert_to(torch.float32)
    sd = {k: v.float() for k, v in load_file(ss_stem + ".safetensors").items()}
    missing, unexpected = ss_flow.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    ss_flow.eval().to(dev)

    reso, cin = ss_cfg["resolution"], ss_cfg["in_channels"]
    noise = torch.randn(1, cin, reso, reso, reso).to(dev)
    with torch.no_grad():
        z_s = sampler.sample(ss_flow, noise, cond=cond_512, neg_cond=torch.zeros_like(cond_512),
                             **ss_params, verbose=True, tqdm_desc="Sampling sparse structure").samples
    del ss_flow
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    t0 = stage("sampling sparse structure", t0)

    with open(args.ss_dec + ".json") as f:
        sd_cfg = json.load(f)["args"]
    sd_cfg["use_fp16"] = False
    ss_dec = SparseStructureDecoder(**sd_cfg)
    ss_dec.load_state_dict({k: v.float() for k, v in load_file(args.ss_dec + ".safetensors").items()})
    ss_dec.dtype = torch.float32
    ss_dec.eval().float().to(dev)
    with torch.no_grad():
        decoded = ss_dec(z_s) > 0                       # [1,1,64,64,64]
    del ss_dec
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    ss_res = 32                                         # both supported types use 32
    ratio = decoded.shape[2] // ss_res
    if ratio > 1:
        decoded = F.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int().contiguous()
    L = coords.shape[0]
    print(f"scaffold: {L} voxels at {ss_res}^3 ({100.0*L/ss_res**3:.2f}% occupancy)")
    t0 = stage("decoding occupancy", t0)

    mean = torch.tensor(norm["mean"])[None]
    std = torch.tensor(norm["std"])[None]

    # ── shape SLAT ───────────────────────────────────────────────────────────
    lr_stem = os.path.join(args.models, "slat_flow_img2shape_dit_1_3B_512_bf16")
    flow_lr, cfg_lr = load_flow(lr_stem, dev)
    lr_noise = torch.randn(L, cfg_lr["in_channels"]).to(dev)
    x0 = sp.SparseTensor(feats=lr_noise, coords=coords.to(dev))
    with torch.no_grad():
        slat = sampler.sample(flow_lr, x0, cond=cond_512, neg_cond=torch.zeros_like(cond_512),
                              **slat_params, verbose=True, tqdm_desc="Sampling shape SLat").samples
    slat = slat * std.to(slat.device) + mean.to(slat.device)
    del flow_lr
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    t0 = stage("sampling shape SLAT", t0)

    # ── decoder ──────────────────────────────────────────────────────────────
    resolution = 512 if args.pipeline_type == "512" else 1024
    dec = load_decoder(args.models, resolution, dec_dev)
    margin = float(dec.voxel_margin)

    if args.pipeline_type == "1024_cascade":
        # sample_shape_slat_cascade: upsample the LR scaffold to 512^3
        # candidates, quantize them into the HR flow's 64^3 scaffold under the
        # token budget, then sample with the 1024 model and cond_1024.
        with torch.no_grad():
            up_coords = dec.upsample(slat.to(dec_dev), upsample_times=4)
        print(f"upsample: {up_coords.shape[0]} candidates at 512^3")
        hr_res = resolution
        while True:
            quant = torch.cat([up_coords[:, :1],
                               ((up_coords[:, 1:] + 0.5) / 512 * (hr_res // 16)).int()], dim=1)
            hr_coords = quant.unique(dim=0)
            if hr_coords.shape[0] < args.max_num_tokens or hr_res == 1024:
                break
            hr_res -= 128
        Lhr = hr_coords.shape[0]
        print(f"hr scaffold: {Lhr} voxels at {hr_res//16}^3 (resolution {hr_res})")
        t0 = stage("upsampling scaffold", t0)

        hr_stem = os.path.join(args.models, "slat_flow_img2shape_dit_1_3B_1024_bf16")
        flow_hr, cfg_hr = load_flow(hr_stem, dev)
        hr_noise = torch.randn(Lhr, cfg_hr["in_channels"]).to(dev)
        xh = sp.SparseTensor(feats=hr_noise, coords=hr_coords.to(dev))
        with torch.no_grad():
            slat = sampler.sample(flow_hr, xh, cond=cond_1024, neg_cond=torch.zeros_like(cond_1024),
                                  **slat_params, verbose=True,
                                  tqdm_desc="Sampling shape SLat (HR)").samples
        slat = slat * std.to(slat.device) + mean.to(slat.device)
        del flow_hr
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        resolution = hr_res
        dec.set_resolution(resolution)
        t0 = stage(f"sampling shape SLAT ({hr_res})", t0)

    # Everything above is GPU work that must not have to be repeated if the
    # decode below runs the host out of memory.
    save_slat_state(state_path, slat, resolution, L, args.pipeline_type)

    decode_and_write(dec, slat, resolution, margin, dec_dev, L,
                     args, timings, stage, t0, ss_params, slat_params)


if __name__ == "__main__":
    main()
