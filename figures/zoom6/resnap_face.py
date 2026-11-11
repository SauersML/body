# face spots: ray from the head center along the base-mesh direction (keeps the angular layout of the face spots on the faceless egg head)
import numpy as np, trimesh, pickle
a=np.load('../zoom3/mesh.npz'); b=np.load('mesh.npz'); B=trimesh.Trimesh(b['v'],b['f'],process=False)
R0=pickle.load(open('../zoom3/reg.pkl','rb')); R=pickle.load(open('reg.pkl','rb'))
def hc(v):
    h=v[(v[:,1]>158)&(np.abs(v[:,0])<15)]; return (h.min(0)+h.max(0))/2, h.min(0), h.max(0)
cb,lo,hi=hc(a['v']); cn,lo2,hi2=hc(b['v']); print('base head',cb.round(1),lo.round(1),hi.round(1)); print('neutral head',cn.round(1),lo2.round(1),hi2.round(1))
p0=R0['res']['face']['pos']; d=p0-cb; d/=np.linalg.norm(d,axis=1,keepdims=True)
RI=trimesh.ray.ray_triangle.RayMeshIntersector(B)
loc,ir,it=RI.intersects_location(np.repeat(cn[None],len(d),0),d,multiple_hits=True)
pos=np.zeros_like(p0); tid=np.zeros(len(d),int)
for i in range(len(d)):
    m=np.where(ir==i)[0]; t=((loc[m]-cn)@d[i]); k=m[np.argmax(t)]  # outermost hit (skin)
    pos[i]=loc[k]; tid[i]=it[k]
R['res']['face']['pos']=pos; R['res']['face']['nrm']=B.face_normals[tid]
print('y range base',p0[:,1].min().round(1),p0[:,1].max().round(1),' neutral',pos[:,1].min().round(1),pos[:,1].max().round(1))
pickle.dump(R,open('reg.pkl','wb'))
