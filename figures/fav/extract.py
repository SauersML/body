import numpy as np, json, trimesh
V=[];G={};F=[];cur=None
for l in open('../human/base.obj'):
  if l.startswith('v '): V.append([float(a) for a in l.split()[1:4]])
  elif l.startswith('g '): cur=l.split()[1]
  elif l.startswith('f '):
    ids=[int(t.split('/')[0])-1 for t in l.split()[1:]]
    G.setdefault(cur,set()).update(ids)
    if cur=='body':
      for k in range(1,len(ids)-1): F.append([ids[0],ids[k],ids[k+1]])
V=np.array(V)*10; F=np.array(F)  # cm
J=lambda k: V[list(G[k])].mean(0)
E=J('joint-l-elbow'); Wr=J('joint-l-hand'); S=J('joint-l-shoulder')
T1=J('joint-l-finger-1-1'); L1=J('joint-l-finger-5-1'); tip=J('joint-l-finger-3-4')
a=(Wr-E); Lf=np.linalg.norm(a); a/=Lf
r=T1-L1; r-=a*(r@a); r/=np.linalg.norm(r)
p=np.cross(a,r)
print('forearm cm',Lf,'p',p,'r',r, 'tip dist', np.linalg.norm(tip-E))
loc=(V-E)@np.stack([a,p,-r],1)   # x along arm, y palm, z thumb
# region: x in [-9, Lf+25], radial dist small; tie to upper arm direction for x<0
up=(S-E)/np.linalg.norm(S-E)
def keep(i):
  pass
x=loc[:,0]; rad=np.hypot(loc[:,1],loc[:,2])
d_up=np.linalg.norm(np.cross(V-E,up),axis=1); t_up=(V-E)@up
m=((x>=-1)&(x<Lf+26)&(rad<(np.where(x<Lf,8.5,12))))|((t_up>0)&(t_up<12)&(d_up<8.5))|(np.linalg.norm(V-E,axis=1)<8)
fm=m[F].all(1); Fk=F[fm]
tm=trimesh.Trimesh(loc,Fk,process=False); tm.remove_unreferenced_vertices()
# largest component
cs=tm.split(only_watertight=False); tm=max(cs,key=lambda c:len(c.faces))
print(len(tm.vertices),len(tm.faces), tm.bounds)
from collections import Counter
fe=np.concatenate([tm.faces[:,[0,1]],tm.faces[:,[1,2]],tm.faces[:,[2,0]]]); cnt=Counter(map(tuple,np.sort(fe,1)))
bd=[e for e in fe if cnt[tuple(sorted(e))]==1]
bv=sorted(set(np.array(bd).ravel())); cen=tm.vertices[bv].mean(0)
vv=np.vstack([tm.vertices,cen]); ci=len(vv)-1
ff=np.vstack([tm.faces,[[b,a,ci] for a,b in bd]])
tm=trimesh.Trimesh(vv,ff,process=False); print('capped',len(bd))
for _ in range(2):
  v,f=trimesh.remesh.subdivide_loop(tm.vertices,tm.faces,iterations=1); tm=trimesh.Trimesh(v,f,process=False)
print(len(tm.vertices))
upl=(S-E)@np.stack([a,p,-r],1)/np.linalg.norm(S-E)
json.dump({'v':np.round(tm.vertices,3).ravel().tolist(),'f':tm.faces.ravel().tolist(),'L':Lf,'up':upl.tolist()},open('arm.json','w'))
