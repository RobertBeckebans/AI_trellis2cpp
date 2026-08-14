#!/usr/bin/env python3
"""Reference dumps for the 1024_cascade high-resolution geometry stage.

Produces dumps/reference_cascade.gguf with the full HR chain so the C++ port can
be validated stage by stage:

  cond_512, cond_1024        the two DINOv3 conds (embedded → self-contained test)
  coords32   [L,4]           32^3 scaffold (from the SS reference latent)
  lr_noise   [L,32]          LR sampling noise (seed 4321)
  lr_slat    [L,32]          512-model sampler output, denormalized
  up_coords  [Nup,4]         decoder.upsample(lr_slat, upsample_times=4) → 512^3
  hr_coords  [Lhr,4]         quantized+unique → 64^3 (the HR flow scaffold)
  hr_coords_1536 [L',4]      the same candidates through the 1536 token budget
  hr_resolution_1536 [1]     the resolution that budget settled on (1024..1536)
  hr_noise   [Lhr,32]        HR sampling noise (seed 5678)
  hr_flow_t500_out [Lhr,32]  1024-model flow forward at t=500 on hr_coords
  hr_slat    [Lhr,32]        1024-model sampler output, denormalized
  lvl{i}.*, out7, out_coords per-level decode taps at resolution 1024 (→ 1024^3)

The reference is generated with TF32 disabled (ref_common.setup) so the golden
values are true fp32. Run inside the container (see scripts/refgen.sh).
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_common  # noqa: E402

ref_common.setup()  # TF32 off + sdpa sparse attention + pure-torch sparse conv

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def read_dinodata(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"DINOCOND", "bad magic"
        _, _, ndim = struct.unpack("<III", f.read(12))
        shape = struct.unpack("<%dI" % ndim, f.read(4 * ndim))
        arr = np.frombuffer(f.read(), dtype="<f4").reshape(shape)
    return arr


def load_ss_sample_latent(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"SSSAMP01", "bad magic"
        R, cin, lkv, cctx, steps = struct.unpack("<5i", f.read(20))
        n = cin * R * R * R
        f.seek(-(n * 4), os.SEEK_END)
        z = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(1, cin, R, R, R)
    return torch.from_numpy(z.copy())


def load_flow(stem, dev):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "ckpts"))
    ap.add_argument("--ss-dec", default=os.path.join(ref_common.MODELS, "TRELLIS-image-large",
                                                     "ckpts", "ss_dec_conv3d_16l8_fp16"))
    ap.add_argument("--cond-512", default=os.path.join(ref_common.DUMPS, "fixture.dinodata"))
    ap.add_argument("--cond-1024", default=os.path.join(ref_common.DUMPS, "fixture_1024.dinodata"))
    ap.add_argument("--ss-latent", default=os.path.join(ref_common.REPO, "tests", "ss_sample_ref.bin"))
    ap.add_argument("--pipeline-json", default=os.path.join(ref_common.MODELS, "TRELLIS.2-4B", "pipeline.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--t", type=float, default=500.0)
    ap.add_argument("--lr-seed", type=int, default=4321)
    ap.add_argument("--hr-seed", type=int, default=5678)
    ap.add_argument("--lr-resolution", type=int, default=512)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--resolution-1536", type=int, default=1536,
                    help="second tier captured for the token-budget reference")
    ap.add_argument("--max-num-tokens", type=int, default=49152,
                    help="HR token budget the reduction loop is run against")
    ap.add_argument("--skip-hr-sampler", action="store_true",
                    help="stop after the coordinate captures: no HR flow forward, no HR "
                         "sampler, no final decode. hr_coords_1536 / hr_resolution_1536 "
                         "depend only on up_coords, which exists before the HR chain, so "
                         "this produces the 1536 token-budget reference at a fraction of "
                         "the cost. The omitted tensors are absent from the dump, and "
                         "tests/test_cascade.cpp reports them as not present rather than "
                         "failing.")
    ap.add_argument("--out", default=os.path.join(ref_common.DUMPS, "reference_cascade.gguf"))
    args = ap.parse_args()

    from safetensors.torch import load_file
    from trellis2.models.sparse_structure_vae import SparseStructureDecoder
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2.modules import sparse as sp

    dev = torch.device(args.device)
    caps = {}

    cond_512 = torch.from_numpy(read_dinodata(args.cond_512).copy()).float().to(dev)
    cond_1024 = torch.from_numpy(read_dinodata(args.cond_1024).copy()).float().to(dev)
    caps["cond_512"] = cond_512[0]
    caps["cond_1024"] = cond_1024[0]

    with open(args.pipeline_json) as f:
        pj = json.load(f)["args"]
    norm = pj["shape_slat_normalization"]
    sampler_params = pj["shape_slat_sampler"]["params"]
    mean = torch.tensor(norm["mean"], device=dev)[None]
    std = torch.tensor(norm["std"], device=dev)[None]
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)

    # ── 32^3 scaffold from the SS reference latent ───────────────────────────
    with open(args.ss_dec + ".json") as f:
        ss_cfg = json.load(f)["args"]
    ss_cfg["use_fp16"] = False
    ss_dec = SparseStructureDecoder(**ss_cfg)
    ss_dec.load_state_dict({k: v.float() for k, v in load_file(args.ss_dec + ".safetensors").items()})
    ss_dec.dtype = torch.float32
    ss_dec.eval().float().to(dev)
    z_s = load_ss_sample_latent(args.ss_latent).to(dev)
    with torch.no_grad():
        occ = ss_dec(z_s) > 0
    occ = (F.max_pool3d(occ.float(), 2, 2, 0) > 0.5)
    coords = torch.argwhere(occ)[:, [0, 2, 3, 4]].int().contiguous()
    L = coords.shape[0]
    caps["coords32"] = coords.float()
    print(f"scaffold: {L} voxels at 32^3")
    del ss_dec

    # ── LR flow: sample with the 512 model + cond_512, denormalize ───────────
    flow_lr, cfg_lr = load_flow(os.path.join(args.models, "slat_flow_img2shape_dit_1_3B_512_bf16"), dev)
    g = torch.Generator().manual_seed(args.lr_seed)
    lr_noise = torch.randn(L, cfg_lr["in_channels"], generator=g).to(dev)
    caps["lr_noise"] = lr_noise
    x0 = sp.SparseTensor(feats=lr_noise.clone(), coords=coords.to(dev))
    with torch.no_grad():
        lr_slat = sampler.sample(flow_lr, x0, cond=cond_512, neg_cond=torch.zeros_like(cond_512),
                                 **sampler_params, verbose=True).samples
    lr_slat = lr_slat * std + mean
    caps["lr_slat"] = lr_slat.feats
    print(f"lr_slat: {lr_slat.feats.shape} mean={lr_slat.feats.mean().item():.5f}")
    del flow_lr
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ── shape decoder: upsample(×4) → 512^3 candidate coords ─────────────────
    dstem = os.path.join(args.models, "shape_dec_next_dc_f16c32_fp16")
    with open(dstem + ".json") as f:
        dcfg = json.load(f)["args"]
    dcfg.pop("use_fp16", None)
    dcfg.pop("resolution", None)
    dec = FlexiDualGridVaeDecoder(resolution=args.resolution, use_fp16=False, **dcfg)
    dec.load_state_dict({k: v.float() for k, v in load_file(dstem + ".safetensors").items()})
    # The decoder runs on CPU: the 1024^3 expansion materializes millions of
    # voxels through the pure-torch sparse conv and would OOM the 16 GB GPU (and
    # it is exactly what the C++ port runs on CPU). Flows stay on GPU.
    dec.eval().float().cpu()

    with torch.no_grad():
        up_coords = dec.upsample(lr_slat.cpu(), upsample_times=4)   # [Nup, 4] at 512^3
    caps["up_coords"] = up_coords.float()
    print(f"upsample: {up_coords.shape[0]} candidate coords at {args.lr_resolution}^3")

    # ── quantize + unique → 64^3 HR scaffold (verbatim pipeline formula) ─────
    hr_res = args.resolution
    quant_coords = torch.cat([
        up_coords[:, :1],
        ((up_coords[:, 1:] + 0.5) / args.lr_resolution * (hr_res // 16)).int(),
    ], dim=1)
    hr_coords = quant_coords.unique(dim=0)
    Lhr = hr_coords.shape[0]
    caps["hr_coords"] = hr_coords.float()
    print(f"hr scaffold: {Lhr} voxels at {hr_res // 16}^3")

    # ── the same candidates through the 1536 tier's token budget ────────────
    # Pipeline lines 411-424: start at the requested resolution, quantize,
    # dedup, and step down by 128 while the scaffold would reach
    # max_num_tokens, with 1024 as the floor. Pure coordinate work — no extra
    # model runs — so it costs nothing to capture alongside the 1024 chain and
    # gives the C++ token loop a real reference instead of a synthetic one.
    hr_res_1536 = args.resolution_1536
    while True:
        q1536 = torch.cat([
            up_coords[:, :1],
            ((up_coords[:, 1:] + 0.5) / args.lr_resolution * (hr_res_1536 // 16)).int(),
        ], dim=1)
        coords_1536 = q1536.unique(dim=0)
        if coords_1536.shape[0] < args.max_num_tokens or hr_res_1536 == 1024:
            break
        hr_res_1536 -= 128
    caps["hr_coords_1536"] = coords_1536.float()
    caps["hr_resolution_1536"] = torch.tensor([float(hr_res_1536)])
    print(f"1536 budget: {coords_1536.shape[0]} voxels at {hr_res_1536 // 16}^3 "
          f"(resolution {hr_res_1536}, budget {args.max_num_tokens})")

    if args.skip_hr_sampler:
        # Everything below needs the 1024 flow model and the 1024^3 decode; the
        # coordinate captures above do not. Omitting them leaves hr_noise,
        # hr_flow_t500_out, hr_slat, out7 and out_coords out of the dump.
        print("--skip-hr-sampler: stopping after the coordinate captures "
              "(no HR flow, no HR sampler, no 1024^3 decode)")
    else:
        # ── HR flow: forward @ t=500 + full sampler with 1024 model + cond_1024 ──
        flow_hr, cfg_hr = load_flow(os.path.join(args.models, "slat_flow_img2shape_dit_1_3B_1024_bf16"), dev)
        g = torch.Generator().manual_seed(args.hr_seed)
        hr_noise = torch.randn(Lhr, cfg_hr["in_channels"], generator=g).to(dev)
        caps["hr_noise"] = hr_noise
        xh = sp.SparseTensor(feats=hr_noise.clone(), coords=hr_coords.to(dev))
        with torch.no_grad():
            out = flow_hr(xh, torch.tensor([args.t], device=dev), cond_1024)
        caps["hr_flow_t500_out"] = out.feats
        print(f"hr flow t={args.t}: l2={out.feats.norm().item():.4f}")

        xh0 = sp.SparseTensor(feats=hr_noise.clone(), coords=hr_coords.to(dev))
        with torch.no_grad():
            hr_slat = sampler.sample(flow_hr, xh0, cond=cond_1024, neg_cond=torch.zeros_like(cond_1024),
                                     **sampler_params, verbose=True).samples
        hr_slat = hr_slat * std + mean
        caps["hr_slat"] = hr_slat.feats
        print(f"hr_slat: {hr_slat.feats.shape} mean={hr_slat.feats.mean().item():.5f}")
        del flow_hr
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        # ── final decode of the HR slat at resolution 1024, level by level ───────
        # Only the final 7-channel output is kept: the per-level intermediates are
        # multiple GB at 1024^3 (and the decoder's level logic is already validated
        # exactly at the 512 tier). We run the base SparseUnetVaeDecoder forward
        # manually to get the raw 7 channels — dec(...) would run the FDG mesh
        # conversion (stubbed o_voxel). Running the decode on CPU is host-RAM heavy.
        dec.set_resolution(args.resolution)
        hr_slat_cpu = hr_slat.cpu()
        with torch.no_grad():
            h = dec.from_latent(hr_slat_cpu.float())
            for i, res in enumerate(dec.blocks):
                for j, block in enumerate(res):
                    if i < len(dec.blocks) - 1 and j == len(res) - 1:
                        h, _sub = block(h)
                    else:
                        h = block(h)
                print(f"decode level {i}: {h.feats.shape[0]} voxels x {h.feats.shape[1]} ch")
            hn = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            out7 = dec.output_layer(hn)
        caps["out7"] = out7.feats
        caps["out_coords"] = out7.coords.float()
        print(f"out7: {out7.feats.shape}")

    import gguf
    writer = gguf.GGUFWriter(args.out, "reference")
    manifest = {"shapes": {}, "atol": 2e-3, "rtol": 2e-3,
                "lr_resolution": args.lr_resolution, "resolution": args.resolution,
                "skip_hr_sampler": bool(args.skip_hr_sampler)}
    for name, t in caps.items():
        a = t.detach().cpu().float().numpy()
        manifest["shapes"][name] = list(a.shape)
        writer.add_tensor(name, np.ascontiguousarray(a.reshape(-1), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with open(os.path.join(ref_common.DUMPS, "manifest_cascade.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes), {len(caps)} tensors")


if __name__ == "__main__":
    main()
