"""Why are whole regions still back-wound?

Splits the mesh two ways and reports both, which distinguishes the two possible
causes:
  * vertex-connected COMPONENTS with a bad global sign  -> orient_faces step 2
    picked wrong (e.g. open sheets where signed volume is meaningless);
  * consistently-wound PATCHES inside one component     -> the seam is a
    frustrated edge, i.e. the normal sign field is wrong, not the components.
"""
import json
import struct
import sys
from array import array

path = sys.argv[1]
blob = open(path, "rb").read()
off = 12
chunks = []
while off < len(blob):
    clen, ctype = struct.unpack_from("<II", blob, off)
    chunks.append((ctype, blob[off + 8 : off + 8 + clen]))
    off += 8 + clen
gltf = json.loads(chunks[0][1].decode("utf-8").rstrip("\x00 "))
binc = chunks[1][1]
CT = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read(i):
    a = gltf["accessors"][i]
    bv = gltf["bufferViews"][a["bufferView"]]
    fmt, sz = CT[a["componentType"]]
    n = NC[a["type"]] * a["count"]
    o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    arr = array(fmt)
    arr.frombytes(binc[o : o + n * sz])
    return arr


prim = gltf["meshes"][0]["primitives"][0]
pos = read(prim["attributes"]["POSITION"])
idx = read(prim["indices"])
nv = len(pos) // 3
nt = len(idx) // 3
print("verts", nv, "tris", nt)

# ---------------------------------------------------------------- union-find
par = list(range(max(nv, nt)))


def find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def uni(p, a, b):
    a, b = find(p, a), find(p, b)
    if a != b:
        p[a] = b


# (A) vertex-connected components
pv = list(range(nv))
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    uni(pv, a, b)
    uni(pv, b, c)
comp = {}
for t in range(nt):
    r = find(pv, idx[3 * t])
    comp[r] = comp.get(r, 0) + 1
print("vertex-connected components:", len(comp))
for r, n in sorted(comp.items(), key=lambda kv: -kv[1])[:8]:
    print("   %8d tris" % n)

# (B) patches: faces joined only across edges they wind CONSISTENTLY
#     (one face traverses the shared edge u->v, the other v->u)
half = {}
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    for u, v in ((a, b), (b, c), (c, a)):
        k = (u, v) if u < v else (v, u)
        half.setdefault(k, []).append((t, 1 if u < v else -1))

pf = list(range(nt))
consistent = inconsistent = nonmanifold = 0
for k, lst in half.items():
    if len(lst) != 2:
        nonmanifold += 1
        continue
    (t0, d0), (t1, d1) = lst
    if d0 != d1:
        uni(pf, t0, t1)
        consistent += 1
    else:
        inconsistent += 1
print("manifold edges: %d consistent, %d back-wound seam; %d non-manifold" % (consistent, inconsistent, nonmanifold))

patch = {}
for t in range(nt):
    r = find(pf, t)
    patch[r] = patch.get(r, 0) + 1
print("consistently-wound patches:", len(patch))
tot = 0
for r, n in sorted(patch.items(), key=lambda kv: -kv[1])[:10]:
    tot += n
    print("   %8d tris (%.1f%%)" % (n, 100.0 * n / nt))
print("   top-10 cover %.1f%% of all tris" % (100.0 * tot / nt))

# signed volume per big patch, to see whether a patch is even closed enough
# for the volume sign test orient_faces uses
big = [r for r, n in sorted(patch.items(), key=lambda kv: -kv[1])[:6]]
for r in big:
    sx = sy = sz = cnt = 0
    for t in range(nt):
        if find(pf, t) != r:
            continue
        for k in range(3):
            v = idx[3 * t + k]
            sx += pos[3 * v]
            sy += pos[3 * v + 1]
            sz += pos[3 * v + 2]
            cnt += 1
    ox, oy, oz = sx / cnt, sy / cnt, sz / cnt
    vol = 0.0
    bnd = 0
    for t in range(nt):
        if find(pf, t) != r:
            continue
        a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
        ax, ay, az = pos[3 * a] - ox, pos[3 * a + 1] - oy, pos[3 * a + 2] - oz
        bx, by, bz = pos[3 * b] - ox, pos[3 * b + 1] - oy, pos[3 * b + 2] - oz
        cx, cy, cz = pos[3 * c] - ox, pos[3 * c + 1] - oy, pos[3 * c + 2] - oz
        vol += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    print("patch %8d tris  signed volume %+.6f" % (patch[r], vol / 6.0))
