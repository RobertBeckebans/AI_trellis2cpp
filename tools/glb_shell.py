"""Was the removed half an inner shell, or is the cleaned mesh now holed?

  glb_shell.py A.glb meshA B.glb meshB

Compares openness (boundary edges per triangle) before and after, and the
radial distribution of the removed set against the surviving one: an inner
shell sits systematically closer to the model axis, a holed surface does not.
"""
import json
import struct
import sys
from array import array
from collections import Counter

CT = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def load(path, mesh_index):
    blob = open(path, "rb").read()
    off = 12
    chunks = []
    while off < len(blob):
        clen, ctype = struct.unpack_from("<II", blob, off)
        chunks.append((ctype, blob[off + 8 : off + 8 + clen]))
        off += 8 + clen
    g = json.loads(chunks[0][1].decode("utf-8").rstrip("\x00 "))
    binc = chunks[1][1]

    def read(i):
        a = g["accessors"][i]
        bv = g["bufferViews"][a["bufferView"]]
        fmt, sz = CT[a["componentType"]]
        n = NC[a["type"]] * a["count"]
        o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        arr = array(fmt)
        arr.frombytes(binc[o : o + n * sz])
        return arr

    p = g["meshes"][mesh_index]["primitives"][0]
    return read(p["attributes"]["POSITION"]), read(p["indices"])


def weld(pos, idx):
    """Blender re-splits vertices per UV island; weld by position so the
    topology numbers are comparable between the two files."""
    m = {}
    remap = []
    for v in range(len(pos) // 3):
        k = (round(pos[3 * v] * 1e6), round(pos[3 * v + 1] * 1e6), round(pos[3 * v + 2] * 1e6))
        r = m.setdefault(k, len(m))
        remap.append(r)
    return remap, len(m)


def report(name, pos, idx):
    remap, nw = weld(pos, idx)
    nt = len(idx) // 3
    e = Counter()
    for t in range(nt):
        a, b, c = remap[idx[3 * t]], remap[idx[3 * t + 1]], remap[idx[3 * t + 2]]
        for u, v in ((a, b), (b, c), (c, a)):
            e[(u, v) if u < v else (v, u)] += 1
    b1 = sum(1 for v in e.values() if v == 1)
    b3 = sum(1 for v in e.values() if v > 2)
    print(
        "%-8s %6d tris  %6d welded verts  %6d edges: %5d boundary (%.1f%%), %4d non-manifold"
        % (name, nt, nw, len(e), b1, 100.0 * b1 / len(e), b3)
    )
    return remap


QUANT = 100000.0


def centroids(pos, idx):
    return [
        (
            round((pos[3 * idx[3 * t]] + pos[3 * idx[3 * t + 1]] + pos[3 * idx[3 * t + 2]]) / 3.0 * QUANT),
            round((pos[3 * idx[3 * t] + 1] + pos[3 * idx[3 * t + 1] + 1] + pos[3 * idx[3 * t + 2] + 1]) / 3.0 * QUANT),
            round((pos[3 * idx[3 * t] + 2] + pos[3 * idx[3 * t + 1] + 2] + pos[3 * idx[3 * t + 2] + 2]) / 3.0 * QUANT),
        )
        for t in range(len(idx) // 3)
    ]


pa, ia = load(sys.argv[1], int(sys.argv[2]))
pb, ib = load(sys.argv[3], int(sys.argv[4]))
print("openness before and after the hand cleanup:")
report("before", pa, ia)
report("after", pb, ib)

ca, cb = centroids(pa, ia), centroids(pb, ib)
cnt = Counter(cb)


def take(key):
    for dx in (0, 1, -1):
        for dy in (0, 1, -1):
            for dz in (0, 1, -1):
                k = (key[0] + dx, key[1] + dy, key[2] + dz)
                if cnt.get(k):
                    cnt[k] -= 1
                    return True
    return False


kept = [t for t in range(len(ca)) if take(ca[t])]
keptset = set(kept)
removed = [t for t in range(len(ca)) if t not in keptset]

# radial profile about the model axis (the figure is upright in Y)
ox = sum(c[0] for c in ca) / len(ca) / QUANT
oz = sum(c[2] for c in ca) / len(ca) / QUANT


def radial(ts):
    r = sorted(((ca[t][0] / QUANT - ox) ** 2 + (ca[t][2] / QUANT - oz) ** 2) ** 0.5 for t in ts)
    return r[len(r) // 10], r[len(r) // 2], r[9 * len(r) // 10]


print("\ndistance from the model's vertical axis (10th / 50th / 90th percentile):")
print("  survived %d tris:  %.4f  %.4f  %.4f" % ((len(kept),) + radial(kept)))
print("  removed  %d tris:  %.4f  %.4f  %.4f" % ((len(removed),) + radial(removed)))
