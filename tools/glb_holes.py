"""Boundary loops of a mesh: how many, and how big. An alpha ball of radius r
can enter any opening wider than r, which is what turns a wrap into a membrane."""
import json, struct, sys
from array import array
from collections import Counter, defaultdict
CT = {5121:("B",1),5123:("H",2),5125:("I",4),5126:("f",4)}
NC = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}
blob = open(sys.argv[1],"rb").read(); off=12; ch=[]
while off < len(blob):
    cl,ct = struct.unpack_from("<II",blob,off); ch.append((ct,blob[off+8:off+8+cl])); off += 8+cl
g = json.loads(ch[0][1].decode("utf-8").rstrip("\x00 ")); binc = ch[1][1]
def read(i):
    a=g["accessors"][i]; bv=g["bufferViews"][a["bufferView"]]; f,s=CT[a["componentType"]]
    n=NC[a["type"]]*a["count"]; o=bv.get("byteOffset",0)+a.get("byteOffset",0)
    r=array(f); r.frombytes(binc[o:o+n*s]); return r
mi = int(sys.argv[2]) if len(sys.argv)>2 else 0
p = g["meshes"][mi]["primitives"][0]
pos, idx = read(p["attributes"]["POSITION"]), read(p["indices"])
nt = len(idx)//3
# weld by position first
wid={}; remap=array("i",bytes(4*(len(pos)//3)))
for v in range(len(pos)//3):
    k=(round(pos[3*v]*1e6),round(pos[3*v+1]*1e6),round(pos[3*v+2]*1e6))
    remap[v]=wid.setdefault(k,len(wid))
e=Counter()
for t in range(nt):
    a,b,c = remap[idx[3*t]],remap[idx[3*t+1]],remap[idx[3*t+2]]
    for u,v in ((a,b),(b,c),(c,a)):
        e[(u,v) if u<v else (v,u)] += 1
bnd=[k for k,v in e.items() if v==1]
print("welded verts %d, tris %d, boundary edges %d" % (len(wid), nt, len(bnd)))
if not bnd:
    print("closed."); raise SystemExit
adj=defaultdict(list)
for u,v in bnd: adj[u].append(v); adj[v].append(u)
seen=set(); loops=[]
for s in adj:
    if s in seen: continue
    stack=[s]; comp=[]
    seen.add(s)
    while stack:
        x=stack.pop(); comp.append(x)
        for y in adj[x]:
            if y not in seen: seen.add(y); stack.append(y)
    loops.append(comp)
wp=[0.0]*(len(wid)*3)
for v in range(len(pos)//3):
    for k in range(3): wp[remap[v]*3+k]=pos[3*v+k]
sizes=[]
for c in loops:
    lo=[min(wp[3*v+k] for v in c) for k in range(3)]
    hi=[max(wp[3*v+k] for v in c) for k in range(3)]
    sizes.append((max(hi[k]-lo[k] for k in range(3)), len(c)))
sizes.sort(reverse=True)
print("boundary loops: %d" % len(sizes))
print("largest openings (bbox extent / vertices):")
for d,n in sizes[:12]: print("   %.5f  (%d verts)" % (d, n))
import statistics
print("median opening extent: %.5f" % statistics.median(d for d,_ in sizes))
alpha = 0.00499*1.163
print("alpha ball radius for detail 0.499%%: %.5f  -> %d of %d openings are wider"
      % (alpha, sum(1 for d,_ in sizes if d > alpha), len(sizes)))
