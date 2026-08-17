#!/usr/bin/env python3
"""Reference dumps for the shape-SLAT stage ("512" pipeline type, geometry only).

Produces dumps/reference_slat.gguf with:

  coords            [L, 4] f32    active voxels at 32^3 (from the SS stage)
  slat_noise        [L, 32]       fixed sampling noise (seed 4321)
  flow_t500_out     [L, 32]       SLAT flow forward at t=500 (f32)
  slat              [L, 32]       full 12-step sampler output, denormalized
  slat_mean/std     [32]          shape_slat_normalization
  lvl{i}.in_coords  [L_i, 4]      decoder level i active voxels
  lvl{i}.pre_up     [L_i, C_i]    features after level i's ConvNeXt blocks
  lvl{i}.subdiv     [L_i, 8]      subdivision logits of level i's up-block
  out7              [L_4, 7]      decoder output (pre split/sigmoid)
  out_coords        [L_4, 4]

The coords come from decoding tests/ss_sample_ref.bin's reference latent, so
the same scaffold is reproducible from the validated C++ SS stages.

Run inside the container: see scripts/refgen.sh.
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_common  # noqa: E402

ref_common.setup()  # sdpa sparse attention + pure-torch sparse conv

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def load_ss_sample_latent(path):
    """Read the z_s reference produced by tests/ref_ss_sample.py (SSSAMP01)."""
    with open(path, "rb") as f:
        assert f.read(8) == b"SSSAMP01", "bad magic"
        R, cin, lkv, cctx, steps = struct.unpack("<5i", f.read(20))
        f.read(4 * 3)  # gs, rescale, rescale_t
        f.read(8)      # seed
        n = cin * R * R * R
        f.seek(-(n * 4), os.SEEK_END)
        z = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(1, cin, R, R, R)
    return torch.from_numpy(z.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "ckpts"))
    ap.add_argument("--ss-dec", default=os.path.join(ref_common.MODELS, "TRELLIS-image-large",
                                                     "ckpts", "ss_dec_conv3d_16l8_fp16"))
    ap.add_argument("--dinodata", default=os.path.join(ref_common.DUMPS, "fixture.dinodata"))
    ap.add_argument("--ss-latent", default=os.path.join(ref_common.REPO, "tests", "ss_sample_ref.bin"))
    ap.add_argument("--pipeline-json", default=os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "pipeline.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--t", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(ref_common.DUMPS, "reference_slat.gguf"))
    args = ap.parse_args()

    from safetensors.torch import load_file
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder
    from trellis2.models.structured_latent_flow import SLatFlowModel
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2.modules import sparse as sp

    dev = torch.device(args.device)
    caps = {}

    # ── coords from the SS stage reference latent ────────────────────────────
    with open(args.ss_dec + ".json") as f:
        ss_cfg = json.load(f)["args"]
    ss_cfg["use_fp16"] = False
    ss_dec = SparseStructureDecoder(**ss_cfg)
    ss_dec.load_state_dict({k: v.float() for k, v in load_file(args.ss_dec + ".safetensors").items()})
    ss_dec.dtype = torch.float32
    ss_dec.eval().float().to(dev)

    z_s = load_ss_sample_latent(args.ss_latent).to(dev)
    with torch.no_grad():
        occ = ss_dec(z_s) > 0                                   # [1,1,64,64,64]
    occ = (F.max_pool3d(occ.float(), 2, 2, 0) > 0.5)            # ss_res 32
    coords = torch.argwhere(occ)[:, [0, 2, 3, 4]].int().contiguous()
    L = coords.shape[0]
    print(f"coords: {L} active voxels at 32^3 "
          f"({100.0 * L / 32**3:.2f}% occupancy)")
    caps["coords"] = coords.float()
    del ss_dec

    # ── conditioning ─────────────────────────────────────────────────────────
    with open(args.dinodata, "rb") as f:
        assert f.read(8) == b"DINOCOND"
        _, _, ndim = struct.unpack("<III", f.read(12))
        shape = struct.unpack("<%dI" % ndim, f.read(4 * ndim))
        cond_np = np.frombuffer(f.read(), dtype="<f4").reshape(shape)
    cond = torch.from_numpy(cond_np.copy()).float().to(dev)
    neg_cond = torch.zeros_like(cond)

    # ── SLAT flow: forward parity point + full sampler ───────────────────────
    stem = os.path.join(args.models, "slat_flow_img2shape_dit_1_3B_512_bf16")
    with open(stem + ".json") as f:
        cfg = json.load(f)["args"]
    cfg.pop("initialization", None)
    cfg.pop("dtype", None)
    flow = SLatFlowModel(**cfg, dtype="float32")
    sd = {k: v.float() for k, v in load_file(stem + ".safetensors").items()}
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    flow.convert_to(torch.float32)
    flow.eval().to(dev)

    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(L, cfg["in_channels"], generator=g).to(dev)
    caps["slat_noise"] = noise

    x = sp.SparseTensor(feats=noise.clone(), coords=coords.to(dev))
    with torch.no_grad():
        out = flow(x, torch.tensor([args.t], device=dev), cond)
    caps["flow_t500_out"] = out.feats
    print(f"flow t={args.t}: out mean={out.feats.mean().item():.6f} "
          f"l2={out.feats.norm().item():.4f}")

    with open(args.pipeline_json) as f:
        pj = json.load(f)["args"]
    norm = pj["shape_slat_normalization"]
    sampler_params = pj["shape_slat_sampler"]["params"]
    print("sampler params:", sampler_params)

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    x0 = sp.SparseTensor(feats=noise.clone(), coords=coords.to(dev))
    with torch.no_grad():
        slat = sampler.sample(flow, x0, cond=cond, neg_cond=neg_cond,
                              **sampler_params, verbose=True).samples
    mean = torch.tensor(norm["mean"], device=dev)[None]
    std = torch.tensor(norm["std"], device=dev)[None]
    slat = slat * std + mean
    caps["slat"] = slat.feats
    caps["slat_mean"] = mean[0]
    caps["slat_std"] = std[0]
    print(f"slat: mean={slat.feats.mean().item():.5f} std={slat.feats.std().item():.5f}")
    del flow
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ── FDG decoder, level by level (mirrors SparseUnetVaeDecoder.forward) ──
    dstem = os.path.join(args.models, "shape_dec_next_dc_f16c32_fp16")
    with open(dstem + ".json") as f:
        dcfg = json.load(f)["args"]
    dcfg.pop("use_fp16", None)
    dcfg.pop("resolution", None)
    dec = FlexiDualGridVaeDecoder(resolution=args.resolution, use_fp16=False, **dcfg)
    dec.load_state_dict({k: v.float() for k, v in load_file(dstem + ".safetensors").items()})
    dec.eval().float().to(dev)

    # The last up-block's SparseChannel2Spatial runs exactly twice — on the
    # conv1 output, then on the skip source — and those two tensors are what the
    # C++ port hands from one level to the next (it splits this block in half,
    # where the reference keeps it whole). Capturing them here is what lets the
    # span between lvl{n-2}.pre_up and lvl{n-1}.pre_up be bisected at all.
    updown_caps = []
    up_block = dec.blocks[len(dec.blocks) - 2][-1]
    hook = up_block.updown.register_forward_hook(
        lambda mod, inp, out: updown_caps.append(out.feats.detach())
    )
    # norm2 and conv2 of the same block, so the last two steps of
    # `conv2(silu(norm2(h))) + skip` can be checked one at a time rather than
    # as a sum.
    stage_caps = {}
    hooks2 = [
        up_block.norm2.register_forward_hook(
            lambda m, i, o: stage_caps.setdefault("norm2", o.detach())),
        up_block.conv2.register_forward_hook(
            lambda m, i, o: stage_caps.setdefault("conv2", o.feats.detach())),
    ]

    with torch.no_grad():
        h = dec.from_latent(slat.float())
        for i, res in enumerate(dec.blocks):
            caps[f"lvl{i}.in_coords"] = h.coords.float()
            for j, block in enumerate(res):
                if i < len(dec.blocks) - 1 and j == len(res) - 1:
                    caps[f"lvl{i}.pre_up"] = h.feats
                    h, sub = block(h)
                    caps[f"lvl{i}.subdiv"] = sub.feats
                else:
                    h = block(h)
            print(f"level {i}: {h.feats.shape[0]} voxels x {h.feats.shape[1]} ch")
        # The finest level's features, i.e. the input to the final layer_norm +
        # output_layer. The loop above only captures pre_up for levels that have
        # an up-block (i < n-1), so this one was missing — which left the whole
        # span between lvl{n-2}.pre_up and out7 unbisectable. See
        # docs/progress/rocm-native-reference_3-slat-dump-and-out7.md.
        caps[f"lvl{len(dec.blocks) - 1}.pre_up"] = h.feats
        hn = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        out7 = dec.output_layer(hn)
    hook.remove()
    for h_ in hooks2:
        h_.remove()
    for k_, v_ in stage_caps.items():
        caps[f"lvl{len(dec.blocks) - 1}.{k_}"] = v_
    if len(updown_caps) == 2:
        caps[f"lvl{len(dec.blocks) - 1}.in_hch"] = updown_caps[0]
        caps[f"lvl{len(dec.blocks) - 1}.in_xch"] = updown_caps[1]
    else:
        print(f"WARNING: updown hook fired {len(updown_caps)}x, expected 2 — "
              f"in_hch/in_xch not captured")
    caps["out7"] = out7.feats
    caps["out_coords"] = out7.coords.float()
    print(f"out7: {out7.feats.shape}, offsets mean={torch.sigmoid(out7.feats[:, 0:3]).mean().item():.4f}, "
          f"intersected frac={(out7.feats[:, 3:6] > 0).float().mean().item():.4f}")

    import gguf
    writer = gguf.GGUFWriter(args.out, "reference")
    manifest = {"shapes": {}, "atol": 2e-3, "rtol": 2e-3}
    for name, t in caps.items():
        a = t.detach().cpu().float().numpy()
        manifest["shapes"][name] = list(a.shape)
        writer.add_tensor(name, np.ascontiguousarray(a.reshape(-1), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with open(os.path.join(ref_common.DUMPS, "manifest_slat.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes), {len(caps)} tensors")


if __name__ == "__main__":
    main()
