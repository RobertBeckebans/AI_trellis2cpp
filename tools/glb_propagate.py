"""How large do patches get under proper orientation propagation?

Phase 1 of the planned fix: BFS over face adjacency through *manifold* edges
only (exactly 2 faces), flipping each face so the two traverse the shared edge
in opposite directions. Every such patch is consistent by construction, so the
only thing that can separate two patches afterwards is a non-manifold edge.

Reports the resulting patch sizes, the odd-cycle conflicts BFS could not
satisfy, and each big patch's signed volume about its own centroid -- i.e.
whether a per-patch volume test alone would already settle the sign.
"""
import json
import struct
import sys
from array import array
from collections import deque

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

# ---- edge -> (face, direction) ------------------------------------------
half = {}
for t in range(nt):
    a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
    for u, v in ((a, b), (b, c), (c, a)):
        if u < v:
            half.setdefault((u, v), []).append((t, 1))
        else:
            half.setdefault((v, u), []).append((t, -1))

# manifold adjacency: face -> [(other face, must_be_opposite_dir)]
adj = [[] for _ in range(nt)]
nonman = []
for k, lst in half.items():
    if len(lst) == 2:
        (t0, d0), (t1, d1) = lst
        adj[t0].append((t1, d0, d1))
        adj[t1].append((t0, d1, d0))
    elif len(lst) > 2:
        nonman.append(lst)
print("manifold edges", sum(1 for v in half.values() if len(v) == 2), " non-manifold", len(nonman))

# ---- BFS orientation propagation ----------------------------------------
flip = array("b", bytes(nt))  # 0 = keep, 1 = swap
seen = array("b", bytes(nt))
patch = array("i", bytes(4 * nt))
conflicts = 0
patches = []
pid = 0
for s in range(nt):
    if seen[s]:
        continue
    seen[s] = 1
    patch[s] = pid
    q = deque([s])
    n = 0
    while q:
        f = q.popleft()
        n += 1
        ff = flip[f]
        for g, df, dg in adj[f]:
            # after flipping, f traverses the edge in direction df*(1-2*ff);
            # g must traverse it the other way
            want = 0 if (df * (1 - 2 * ff)) != dg else 1
            if not seen[g]:
                seen[g] = 1
                flip[g] = want
                patch[g] = pid
                q.append(g)
            elif flip[g] != want:
                conflicts += 1
    patches.append(n)
    pid += 1

patches.sort(reverse=True)
print("odd-cycle conflicts on manifold edges:", conflicts // 2)
print("patches after propagation:", len(patches))
for n in patches[:10]:
    print("   %8d tris (%.2f%%)" % (n, 100.0 * n / nt))
print("   top-10 cover %.2f%%" % (100.0 * sum(patches[:10]) / nt))

# ---- signed volume per patch, about its own centroid ---------------------
big = {}
for t in range(nt):
    big[patch[t]] = big.get(patch[t], 0) + 1
order = sorted(big.items(), key=lambda kv: -kv[1])[:6]
for r, n in order:
    sx = sy = sz = cnt = 0
    for t in range(nt):
        if patch[t] != r:
            continue
        for k in range(3):
            v = idx[3 * t + k]
            sx += pos[3 * v]
            sy += pos[3 * v + 1]
            sz += pos[3 * v + 2]
            cnt += 1
    ox, oy, oz = sx / cnt, sy / cnt, sz / cnt
    vol = 0.0
    for t in range(nt):
        if patch[t] != r:
            continue
        a, b, c = idx[3 * t], idx[3 * t + 1], idx[3 * t + 2]
        if flip[t]:
            b, c = c, b
        ax, ay, az = pos[3 * a] - ox, pos[3 * a + 1] - oy, pos[3 * a + 2] - oz
        bx, by, bz = pos[3 * b] - ox, pos[3 * b + 1] - oy, pos[3 * b + 2] - oz
        cx, cy, cz = pos[3 * c] - ox, pos[3 * c + 1] - oy, pos[3 * c + 2] - oz
        vol += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    print("patch %8d tris  signed volume %+.6f" % (n, vol / 6.0))
