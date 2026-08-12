"""Quick structural/statistical dump of a .glb produced by write_vertex_glb."""
import json
import struct
import sys
from array import array

path = sys.argv[1]
blob = open(path, "rb").read()
magic, ver, total = struct.unpack_from("<III", blob, 0)
off = 12
chunks = []
while off < len(blob):
    clen, ctype = struct.unpack_from("<II", blob, off)
    chunks.append((ctype, blob[off + 8 : off + 8 + clen]))
    off += 8 + clen + ((4 - clen % 4) % 4) * 0
gltf = json.loads(chunks[0][1].decode("utf-8").rstrip("\x00 "))
binc = chunks[1][1]
print("chunks:", [(hex(c[0]), len(c[1])) for c in chunks])

CT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
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
att = prim["attributes"]
pos = read(att["POSITION"])
nrm = read(att["NORMAL"])
col = read(att["COLOR_0"])
idx = read(prim["indices"])
nv = len(pos) // 3
nt = len(idx) // 3
print("verts", nv, "tris", nt)

# --- geometry sanity -------------------------------------------------------
bad_nrm = 0
zero_nrm = 0
nan = 0
for i in range(nv):
    x, y, z = nrm[3 * i], nrm[3 * i + 1], nrm[3 * i + 2]
    l2 = x * x + y * y + z * z
    if l2 != l2:
        nan += 1
    elif l2 < 1e-8:
        zero_nrm += 1
    elif abs(l2 - 1.0) > 0.05:
        bad_nrm += 1
print("normals: nan", nan, "zero", zero_nrm, "unnormalized", bad_nrm)

oob = sum(1 for v in idx if v >= nv)
print("indices out of range:", oob)

# degenerate triangles (repeated index)
deg = 0
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    if a == b or b == c or a == c:
        deg += 1
print("degenerate tris:", deg)

# --- duplicate positions (welding check) ----------------------------------
seen = {}
dups = 0
for i in range(nv):
    k = (pos[3 * i], pos[3 * i + 1], pos[3 * i + 2])
    if k in seen:
        dups += 1
    else:
        seen[k] = i
print("exactly duplicated positions:", dups)

# --- colour speckle detection ---------------------------------------------
# average neighbour colour vs own colour, luminance only
lum = array("f", bytes(4 * nv))
for i in range(nv):
    lum[i] = (0.2126 * col[4 * i] + 0.7152 * col[4 * i + 1] + 0.0722 * col[4 * i + 2]) / 65535.0
nsum = array("f", bytes(4 * nv))
ncnt = array("i", bytes(4 * nv))
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    for u, v in ((a, b), (b, c), (c, a)):
        nsum[u] += lum[v]
        ncnt[u] += 1
        nsum[v] += lum[u]
        ncnt[v] += 1
spike = 0
big = 0
for i in range(nv):
    if ncnt[i]:
        d = lum[i] - nsum[i] / ncnt[i]
        if abs(d) > 0.25:
            spike += 1
        if abs(d) > 0.5:
            big += 1
print("colour spikes >0.25:", spike, " >0.5:", big, " of", nv)

alpha_lt = sum(1 for i in range(nv) if col[4 * i + 3] < 62258)
print("alpha < 0.95:", alpha_lt)
mn = min(lum)
mx = max(lum)
print("luminance range", mn, mx)
