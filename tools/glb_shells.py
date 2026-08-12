"""Find interior shells: welded connected components and how they nest.

  glb_shells.py file.glb [mesh_index]

The quad-remesh export splits vertices, so components must be found on
position-welded topology. For each component this reports size, bounding box
and whether it is contained inside the largest component's box -- an interior
shell is a closed component sitting wholly inside the outer surface.
"""
import json
import struct
import sys
from array import array
from collections import Counter

CT = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

path = sys.argv[1]
mi = int(sys.argv[2]) if len(sys.argv) > 2 else 0
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


p = g["meshes"][mi]["primitives"][0]
pos = read(p["attributes"]["POSITION"])
idx = read(p["indices"])
nt = len(idx) // 3

# weld by position
wid = {}
remap = array("i", bytes(4 * (len(pos) // 3)))
for v in range(len(pos) // 3):
    k = (round(pos[3 * v] * 1e6), round(pos[3 * v + 1] * 1e6), round(pos[3 * v + 2] * 1e6))
    remap[v] = wid.setdefault(k, len(wid))
nw = len(wid)
wpos = [0.0] * (nw * 3)
for v in range(len(pos) // 3):
    for k in range(3):
        wpos[remap[v] * 3 + k] = pos[3 * v + k]
print("%d file verts -> %d welded, %d tris" % (len(pos) // 3, nw, nt))

par = list(range(nw))


def find(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


for t in range(nt):
    a, b, c = remap[idx[3 * t]], remap[idx[3 * t + 1]], remap[idx[3 * t + 2]]
    for u, v in ((a, b), (b, c)):
        ru, rv = find(u), find(v)
        if ru != rv:
            par[ru] = rv

comp = {}
for t in range(nt):
    r = find(remap[idx[3 * t]])
    comp.setdefault(r, []).append(t)
print("welded connected components:", len(comp))

info = []
for r, ts in comp.items():
    lo = [1e30] * 3
    hi = [-1e30] * 3
    verts = set()
    vol = 0.0
    for t in ts:
        a, b, c = remap[idx[3 * t]], remap[idx[3 * t + 1]], remap[idx[3 * t + 2]]
        verts.update((a, b, c))
        for v in (a, b, c):
            for k in range(3):
                lo[k] = min(lo[k], wpos[3 * v + k])
                hi[k] = max(hi[k], wpos[3 * v + k])
        ax, ay, az = wpos[3 * a], wpos[3 * a + 1], wpos[3 * a + 2]
        bx, by, bz = wpos[3 * b], wpos[3 * b + 1], wpos[3 * b + 2]
        cx, cy, cz = wpos[3 * c], wpos[3 * c + 1], wpos[3 * c + 2]
        vol += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    e = Counter()
    for t in ts:
        a, b, c = remap[idx[3 * t]], remap[idx[3 * t + 1]], remap[idx[3 * t + 2]]
        for u, v in ((a, b), (b, c), (c, a)):
            e[(u, v) if u < v else (v, u)] += 1
    b1 = sum(1 for x in e.values() if x == 1)
    info.append((len(ts), len(verts), lo, hi, vol / 6.0, b1, len(e)))

info.sort(key=lambda x: -x[0])
big = info[0]
print("\n%-8s %-7s %-9s %-11s %-9s %s" % ("tris", "verts", "boundary", "signed vol", "diag", "inside outer box?"))
inside_n = inside_t = 0
for n, nvv, lo, hi, vol, b1, ne in info[:25]:
    diag = sum((hi[k] - lo[k]) ** 2 for k in range(3)) ** 0.5
    ins = all(lo[k] >= big[2][k] - 1e-6 and hi[k] <= big[3][k] + 1e-6 for k in range(3))
    print("%-8d %-7d %-9d %-+11.6f %-9.4f %s" % (n, nvv, b1, vol, diag, "yes" if ins else "-- outer --"))
for n, nvv, lo, hi, vol, b1, ne in info[1:]:
    if all(lo[k] >= big[2][k] - 1e-6 and hi[k] <= big[3][k] + 1e-6 for k in range(3)):
        inside_n += 1
        inside_t += n
print("\ncomponents fully inside the outer bounding box: %d, %d tris (%.1f%% of the mesh)" % (inside_n, inside_t, 100.0 * inside_t / nt))
