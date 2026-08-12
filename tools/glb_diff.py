"""Which triangles did the hand-cleanup remove, and were they duplicates?

  glb_diff.py A.glb meshA B.glb meshB

Matches triangles between two meshes by centroid, so it survives the
re-indexing a Blender round trip does. Reports whether every removed triangle
had a surviving twin at the same place (a doubled surface, cleanup was
lossless) or not (the cleaned mesh has holes).
"""
import json
import struct
import sys
from array import array

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


QUANT = 100000.0  # 1e-5 of a unit-cube model


def centroids(pos, idx):
    out = []
    for t in range(len(idx) // 3):
        a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
        out.append(
            (
                round((pos[3 * a] + pos[3 * b] + pos[3 * c]) / 3.0 * QUANT),
                round((pos[3 * a + 1] + pos[3 * b + 1] + pos[3 * c + 1]) / 3.0 * QUANT),
                round((pos[3 * a + 2] + pos[3 * b + 2] + pos[3 * c + 2]) / 3.0 * QUANT),
            )
        )
    return out


pa, ia = load(sys.argv[1], int(sys.argv[2]))
pb, ib = load(sys.argv[3], int(sys.argv[4]))
ca, cb = centroids(pa, ia), centroids(pb, ib)
print("A: %d tris   B: %d tris   removed: %d (%.1f%%)" % (len(ca), len(cb), len(ca) - len(cb), 100.0 * (len(ca) - len(cb)) / len(ca)))

# multiset match, tolerant of the +-1 quantisation jitter a round trip adds
from collections import Counter

cnt_b = Counter(cb)


def take(key):
    for dx in (0, 1, -1):
        for dy in (0, 1, -1):
            for dz in (0, 1, -1):
                k = (key[0] + dx, key[1] + dy, key[2] + dz)
                if cnt_b.get(k):
                    cnt_b[k] -= 1
                    return True
    return False


kept = [t for t in range(len(ca)) if take(ca[t])]
keptset = set(kept)
removed = [t for t in range(len(ca)) if t not in keptset]
print("matched in B: %d,  not in B (removed): %d,  unmatched leftovers in B: %d" % (len(kept), len(removed), sum(cnt_b.values())))

# Did every removed triangle have a twin among the survivors?
cnt_kept = Counter(ca[t] for t in kept)


def near_kept(key):
    for dx in (0, 1, -1):
        for dy in (0, 1, -1):
            for dz in (0, 1, -1):
                if cnt_kept.get((key[0] + dx, key[1] + dy, key[2] + dz)):
                    return True
    return False


twinned = sum(1 for t in removed if near_kept(ca[t]))
print("removed triangles that sit exactly on a surviving one: %d of %d (%.1f%%)" % (twinned, len(removed), 100.0 * twinned / max(1, len(removed))))

# And how doubled is the ORIGINAL on its own?
cnt_a = Counter(ca)
dup = sum(v - 1 for v in cnt_a.values() if v > 1)
print("A: triangles sharing a centroid with another A triangle: %d (%.1f%%)" % (dup, 100.0 * dup / len(ca)))
