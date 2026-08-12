"""Are the triangles whose winding opposes their shading normal invisible dust
or a coherent visible region?

Clusters them among themselves and compares their size to the mesh average. A
sliver or a non-manifold junction cannot be shaded consistently by *any*
smoothed normal, so tiny scattered clusters are inherent to the geometry; a
large coherent cluster would mean the orientation is still wrong somewhere.
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
nrm = read(prim["attributes"]["NORMAL"])
idx = read(prim["indices"])
nt = len(idx) // 3

bad = []
area_all = 0.0
area_bad = 0.0
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    ux = pos[3 * b] - pos[3 * a]
    uy = pos[3 * b + 1] - pos[3 * a + 1]
    uz = pos[3 * b + 2] - pos[3 * a + 2]
    vx = pos[3 * c] - pos[3 * a]
    vy = pos[3 * c + 1] - pos[3 * a + 1]
    vz = pos[3 * c + 2] - pos[3 * a + 2]
    gx, gy, gz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    ar = 0.5 * (gx * gx + gy * gy + gz * gz) ** 0.5
    area_all += ar
    sx = nrm[3 * a] + nrm[3 * b] + nrm[3 * c]
    sy = nrm[3 * a + 1] + nrm[3 * b + 1] + nrm[3 * c + 1]
    sz2 = nrm[3 * a + 2] + nrm[3 * b + 2] + nrm[3 * c + 2]
    if gx * sx + gy * sy + gz * sz2 < 0.0:
        bad.append(t)
        area_bad += ar
print("disagreeing tris: %d of %d (%.3f%%)" % (len(bad), nt, 100.0 * len(bad) / nt))
print("their share of surface AREA: %.4f%%" % (100.0 * area_bad / area_all))
print("mean area ratio (bad / all): %.3f" % ((area_bad / max(1, len(bad))) / (area_all / nt)))

# cluster the disagreeing triangles among themselves, via shared edges
inbad = set(bad)
edge = {}
for t in bad:
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    for u, v in ((a, b), (b, c), (c, a)):
        k = (u, v) if u < v else (v, u)
        edge.setdefault(k, []).append(t)
par = {t: t for t in bad}


def find(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


for k, lst in edge.items():
    for i in range(1, len(lst)):
        a, b = find(lst[0]), find(lst[i])
        if a != b:
            par[a] = b
size = {}
for t in bad:
    r = find(t)
    size[r] = size.get(r, 0) + 1
sizes = sorted(size.values(), reverse=True)
print("clusters: %d, largest %d tris, top-5 %s" % (len(sizes), sizes[0] if sizes else 0, sizes[:5]))
print("singletons: %d (%.1f%% of the disagreeing set)" % (sizes.count(1), 100.0 * sizes.count(1) / max(1, len(sizes))))
