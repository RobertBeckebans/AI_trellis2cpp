"""Winding consistency on POSITION-welded topology -- what Blender actually
sees, since its glTF importer merges vertices by position."""
import json, struct, sys
from array import array
from collections import Counter
CT={5121:("B",1),5123:("H",2),5125:("I",4),5126:("f",4)}
NC={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}
blob=open(sys.argv[1],"rb").read(); off=12; ch=[]
while off<len(blob):
    cl,ct=struct.unpack_from("<II",blob,off); ch.append((ct,blob[off+8:off+8+cl])); off+=8+cl
g=json.loads(ch[0][1].decode("utf-8").rstrip("\x00 ")); binc=ch[1][1]
def read(i):
    a=g["accessors"][i]; bv=g["bufferViews"][a["bufferView"]]; f,s=CT[a["componentType"]]
    n=NC[a["type"]]*a["count"]; o=bv.get("byteOffset",0)+a.get("byteOffset",0)
    r=array(f); r.frombytes(binc[o:o+n*s]); return r
mi=int(sys.argv[2]) if len(sys.argv)>2 else 0
p=g["meshes"][mi]["primitives"][0]
pos,idx=read(p["attributes"]["POSITION"]),read(p["indices"])
nt=len(idx)//3
wid={}; remap=array("i",bytes(4*(len(pos)//3)))
for v in range(len(pos)//3):
    k=(round(pos[3*v]*1e6),round(pos[3*v+1]*1e6),round(pos[3*v+2]*1e6))
    remap[v]=wid.setdefault(k,len(wid))
half={}
for t in range(nt):
    f=[remap[idx[3*t]],remap[idx[3*t+1]],remap[idx[3*t+2]]]
    for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
        k=(a,b) if a<b else (b,a)
        half.setdefault(k,[]).append(1 if a<b else -1)
ok=bad=nm=bd=0
for k,l in half.items():
    if len(l)==1: bd+=1
    elif len(l)==2:
        if l[0]!=l[1]: ok+=1
        else: bad+=1
    else: nm+=1
print("%-30s tris %6d  manifold edges: %6d consistent, %5d BACK-WOUND (%.2f%%);  %4d boundary, %3d non-manifold"
      % (sys.argv[3] if len(sys.argv)>3 else "", nt, ok, bad, 100.0*bad/max(1,ok+bad), bd, nm))
