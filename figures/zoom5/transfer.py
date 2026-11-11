# carry the registered spots from the base mesh to the neutral mesh (same faces, same vertex order): barycentric transfer
import numpy as np, trimesh, pickle
a=np.load('../zoom3/mesh.npz'); b=np.load('../nb4/neutral_mesh.npz')
A=trimesh.Trimesh(a['v'],a['f'],process=False); B=trimesh.Trimesh(b['v'],b['f'],process=False)
R=pickle.load(open('../zoom3/reg.pkl','rb'))
for reg,r in R['res'].items():
    p,_,tid=trimesh.proximity.closest_point(A,r['pos']); w=trimesh.triangles.points_to_barycentric(A.triangles[tid],p)
    q=(B.triangles[tid]*w[:,:,None]).sum(1); mv=np.linalg.norm(q-r['pos'],axis=1)
    print(reg,'moved cm median %.2f max %.2f'%(np.median(mv),mv.max()))
    r['pos']=q; r['nrm']=B.face_normals[tid]
pickle.dump(R,open('reg.pkl','wb'))
np.savez('mesh.npz',v=b['v'],f=b['f'])
