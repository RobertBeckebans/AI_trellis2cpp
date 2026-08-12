"""Surface area and enclosed volume of a mesh -- a double-walled membrane has
roughly twice the area of the solid it hugs, and almost none of its volume."""
import json, struct, sys
from array import array
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
ox=sum(pos[3*i] for i in range(len(pos)//3))/(len(pos)//3)
oy=sum(pos[3*i+1] for i in range(len(pos)//3))/(len(pos)//3)
oz=sum(pos[3*i+2] for i in range(len(pos)//3))/(len(pos)//3)
area=0.0; vol=0.0
for t in range(nt):
    a,b,c = idx[3*t],idx[3*t+1],idx[3*t+2]
    ux,uy,uz = pos[3*b]-pos[3*a],pos[3*b+1]-pos[3*a+1],pos[3*b+2]-pos[3*a+2]
    vx,vy,vz = pos[3*c]-pos[3*a],pos[3*c+1]-pos[3*a+1],pos[3*c+2]-pos[3*a+2]
    gx,gy,gz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
    area += 0.5*(gx*gx+gy*gy+gz*gz)**0.5
    ax,ay,az = pos[3*a]-ox,pos[3*a+1]-oy,pos[3*a+2]-oz
    bx,by,bz = pos[3*b]-ox,pos[3*b+1]-oy,pos[3*b+2]-oz
    cx,cy,cz = pos[3*c]-ox,pos[3*c+1]-oy,pos[3*c+2]-oz
    vol += ax*(by*cz-bz*cy)+ay*(bz*cx-bx*cz)+az*(bx*cy-by*cx)
print("%-28s tris %8d   area %.5f   volume %+.6f   V/A^1.5 %+.5f"
      % (sys.argv[3] if len(sys.argv)>3 else "", nt, area, vol/6.0, (vol/6.0)/area**1.5))
