"""Winding / manifoldness check for a .glb produced by write_vertex_glb."""
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


def read(idx):
    a = gltf["accessors"][idx]
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
nv = len(pos) // 3
nt = len(idx) // 3

flipped = 0
tiny = 0
worst = []
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    ax, ay, az = pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]
    bx, by, bz = pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]
    cx, cy, cz = pos[3 * c], pos[3 * c + 1], pos[3 * c + 2]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    gx, gy, gz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    gl = (gx * gx + gy * gy + gz * gz) ** 0.5
    if gl < 1e-14:
        tiny += 1
        continue
    sx = nrm[3 * a] + nrm[3 * b] + nrm[3 * c]
    sy = nrm[3 * a + 1] + nrm[3 * b + 1] + nrm[3 * c + 1]
    sz2 = nrm[3 * a + 2] + nrm[3 * b + 2] + nrm[3 * c + 2]
    d = (gx * sx + gy * sy + gz * sz2) / gl
    if d < 0.0:
        flipped += 1
print("tris", nt, "zero-area", tiny, "winding disagrees with shading normal:", flipped)

# edge manifoldness
edges = {}
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    for u, v in ((a, b), (b, c), (c, a)):
        k = (u, v) if u < v else (v, u)
        edges[k] = edges.get(k, 0) + 1
b1 = sum(1 for v in edges.values() if v == 1)
b2 = sum(1 for v in edges.values() if v == 2)
b3 = sum(1 for v in edges.values() if v > 2)
print("edges", len(edges), "boundary", b1, "manifold", b2, "non-manifold", b3)

# --- global outwardness: signed volume about the mesh centroid --------------
ox = sum(pos[3 * i] for i in range(nv)) / nv
oy = sum(pos[3 * i + 1] for i in range(nv)) / nv
oz = sum(pos[3 * i + 2] for i in range(nv)) / nv
vol = 0.0
outward = 0
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    ax, ay, az = pos[3 * a] - ox, pos[3 * a + 1] - oy, pos[3 * a + 2] - oz
    bx, by, bz = pos[3 * b] - ox, pos[3 * b + 1] - oy, pos[3 * b + 2] - oz
    cx, cy, cz = pos[3 * c] - ox, pos[3 * c + 1] - oy, pos[3 * c + 2] - oz
    vol += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
print("signed volume about centroid:", vol / 6.0)
for i in range(nv):
    if (pos[3 * i] - ox) * nrm[3 * i] + (pos[3 * i + 1] - oy) * nrm[3 * i + 1] + (pos[3 * i + 2] - oz) * nrm[3 * i + 2] > 0:
        outward += 1
print("normals pointing away from centroid: %.1f%%" % (100.0 * outward / nv))
