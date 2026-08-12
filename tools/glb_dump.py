"""Structural dump of any .glb: nodes, meshes, primitives, counts, bounds."""
import json
import struct
import sys

path = sys.argv[1]
blob = open(path, "rb").read()
off = 12
chunks = []
while off < len(blob):
    clen, ctype = struct.unpack_from("<II", blob, off)
    chunks.append((ctype, blob[off + 8 : off + 8 + clen]))
    off += 8 + clen
g = json.loads(chunks[0][1].decode("utf-8").rstrip("\x00 "))
print("file:", path.split("/")[-1], " bin:", len(chunks[1][1]), "bytes")
print("generator:", g["asset"].get("generator"))
for i, n in enumerate(g.get("nodes", [])):
    print(
        "node %d  mesh=%s  name=%s  T=%s  S=%s"
        % (i, n.get("mesh"), n.get("name"), n.get("translation"), n.get("scale"))
    )
for i, m in enumerate(g.get("meshes", [])):
    for j, p in enumerate(m["primitives"]):
        pa = g["accessors"][p["attributes"]["POSITION"]]
        ia = g["accessors"][p["indices"]] if "indices" in p else None
        print(
            "mesh %d prim %d  name=%s  verts=%d tris=%s  attrs=%s  mat=%s"
            % (
                i,
                j,
                m.get("name"),
                pa["count"],
                ia["count"] // 3 if ia else "-",
                ",".join(sorted(p["attributes"])),
                p.get("material"),
            )
        )
        print("        min=%s\n        max=%s" % (pa.get("min"), pa.get("max")))
for i, mt in enumerate(g.get("materials", [])):
    print("material %d  %s  doubleSided=%s" % (i, mt.get("name"), mt.get("doubleSided")))
