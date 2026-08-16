#!/usr/bin/env python3
"""Publish a reference mesh into generations/ so the viewer lists it.

The server persists a finished generation as a directory holding mesh.t2mesh,
the input images and a manifest.json (server/persistence.go). Nothing writes
one from outside the server, which is the last gap in `backend-parity` phase
1c: a PyTorch mesh is only useful for comparison if it sits in the same history
strip as the ROCm, Vulkan and CPU runs.

The manifest is a v1 record with the fields restoreJobs() validates. The device
string is deliberately explicit about what this artefact is — PyTorch geometry
through our extractor — because the viewer shows it on the tile and in the
tooltip, and a mesh labelled like a backend run would be misread as one.

Usage:
    uv run --extra rocm python scripts/ref_publish_generation.py \
        --mesh dumps/einstein_ref512.t2mesh \
        --image assets/einstein.png \
        --info dumps/einstein_ref512.json
"""

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_common  # noqa: E402

from PIL import Image  # noqa: E402


def mesh_counts(path):
    """Read the T2MESH01/02/03 header and check the file length adds up."""
    widths = {b"T2MESH01": 0, b"T2MESH02": 5, b"T2MESH03": 6}
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in widths:
            raise SystemExit(f"{path}: bad mesh magic {magic!r}")
        nv, nt = struct.unpack("<II", f.read(8))
    pbr = widths[magic]
    expected = 16 + nv * 24 + nv * pbr * 4 + nt * 12
    actual = os.path.getsize(path)
    if actual != expected:
        raise SystemExit(f"{path}: {actual} bytes, expected {expected} for "
                         f"{nv} verts / {nt} tris")
    return nv, nt


def thumbnail(image_path, size=96):
    """A data-URL JPEG like the browser posts for a normal generation."""
    img = Image.open(image_path).convert("RGBA")
    flat = Image.new("RGB", img.size, (0, 0, 0))
    flat.paste(img, mask=img.split()[3])
    flat.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    flat.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="the .t2mesh from dual_grid_cli")
    ap.add_argument("--image", required=True, help="the input image the run was given")
    ap.add_argument("--info", default=None, help="ref_generate.py --out-json record")
    ap.add_argument("--store", default=os.path.join(ref_common.REPO, "generations"))
    ap.add_argument("--id", default=None, help="16 hex chars; derived from the inputs if omitted")
    ap.add_argument("--device", default="PyTorch reference, our extraction [PyTorch]")
    ap.add_argument("--quality", default=None, help="defaults to the resolution in --info")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="replace an existing generation of the same id. The id is derived "
                         "from image/seed/quality/device, so a rerun always collides; this "
                         "is the intended way to say 'that one, again'.")
    args = ap.parse_args()

    info = {}
    if args.info:
        with open(args.info) as f:
            info = json.load(f)

    resolution = info.get("resolution", 512)
    quality = args.quality or str(resolution)
    seed = args.seed if args.seed is not None else int(info.get("seed", 0))
    # pipelineForQuality in server/main.go: coarse 0 is not used here, and the
    # numeric tiers are 1/2/3 in declaration order after it.
    pipeline = {"coarse": 1, "512": 2, "1024": 3, "1536": 4}.get(quality, 2)

    nv, nt = mesh_counts(args.mesh)

    if args.id:
        gen_id = args.id
    else:
        h = hashlib.sha256()
        h.update(os.path.basename(args.image).encode())
        h.update(str(seed).encode())
        h.update(quality.encode())
        h.update(args.device.encode())
        gen_id = h.hexdigest()[:16]
    if len(gen_id) != 16:
        raise SystemExit("--id must be exactly 16 characters (the server keys on it)")

    total_ms = sum(t["milliseconds"] for t in info.get("stageTimings", []))
    now = int(time.time() * 1000)
    created = now - total_ms

    manifest = {
        "version": 1,
        "id": gen_id,
        "createdAt": created,
        "startedAt": created,
        "finishedAt": now,
        "durationMs": total_ms,
        "quality": quality,
        "device": args.device,
        "thumbnail": thumbnail(args.image),
        "pipeline": pipeline,
        "seed": seed,
        "steps": info.get("slatSamplerParams", {}).get("steps", 12),
        "textureSteps": 0,
        "guidance": info.get("slatSamplerParams", {}).get("guidance_strength", 7.5),
        "stageTimings": info.get("stageTimings", []),
    }

    # Same commit discipline as persistJob: build a temporary sibling, then
    # rename, so the server never sees a half-written generation.
    os.makedirs(args.store, exist_ok=True)
    final = os.path.join(args.store, gen_id)
    if os.path.exists(final) and not args.force:
        raise SystemExit(f"{final} already exists — rerun with --force to replace it, "
                         f"or pass a different --id")
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    shutil.copyfile(args.mesh, os.path.join(tmp, "mesh.t2mesh"))
    shutil.copyfile(args.image, os.path.join(tmp, "input.img"))
    shutil.copyfile(args.image, os.path.join(tmp, "source.img"))
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Replace by moving the old one aside first and deleting it only once the
    # new one is in place. A delete-then-write would leave nothing at all if the
    # rename failed, and these directories are hundreds of megabytes that took
    # tens of minutes to produce.
    displaced = None
    if os.path.exists(final):
        displaced = final + ".replaced"
        if os.path.exists(displaced):
            shutil.rmtree(displaced)
        os.rename(final, displaced)
    os.rename(tmp, final)
    if displaced:
        shutil.rmtree(displaced)
        print(f"replaced the previous {gen_id}")

    print(f"published {final}")
    print(f"  {nv:,} vertices, {nt:,} triangles, quality {quality}, seed {seed}")
    print(f"  device label: {args.device}")


if __name__ == "__main__":
    main()
