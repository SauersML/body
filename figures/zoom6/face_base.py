# v6: face panel on the real MakeHuman base mesh (zoom3 registration), same color pipeline as colour.py
import numpy as np, trimesh, pickle, json, sys
src=open('colour.py').read()
pre=src.split('M=np.load')[0]; exec(pre)          # states, pca2, procrustes, oklab, lab2lin, SKIN
M=np.load("../zoom3/mesh.npz"); body=trimesh.Trimesh(M['v'],M['f'],process=False); res=pickle.load(open("../zoom3/reg.pkl","rb"))['res']
V=body.vertices; e=body.edges_unique; w=body.edges_unique_length
G=coo_matrix((np.r_[w,w],(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(len(V),)*2).tocsr()
from scipy.spatial import cKDTree
kd=cKDTree(V); reg="face"
S0=json.load(open("spots.json")); ix=REG[reg]; pos=res[reg]['pos']; c32=np.array(S0["face"]["c32"]); cw=np.array(S0["face"]["cw"])  # exactly the wheel coordinates shown in v5
_,vi=kd.query(pos); D=dijkstra(G,indices=vi,limit=12.0)
Ds=D[:,vi]; np.fill_diagonal(Ds,np.inf); nn=Ds.min(1); sig=np.clip(0.9*nn,0.8*np.median(nn),1.6*np.median(nn))
Wt=np.exp(-np.where(np.isinf(D),1e9,D)**2/(2*sig[:,None]**2)); ws=Wt.sum(0); al=1-np.exp(-ws/0.35)
Dd=np.where(np.isinf(D),1e9,D); Wi=Wt/(Dd**2+(1.0*np.median(nn))**2)
for s,c in [("m",c32),("w",cw)]:
    lab=(Wi.T@oklab(c))/np.maximum(Wi.sum(0),1e-12)[:,None]; L=SKIN*(1-al[:,None])+lab*al[:,None]
    (np.clip(lab2lin(L),0,1)**(1/2.2)*255).round().astype(np.uint8).tofile(f"col_faceB_{s}.bin")
V.astype(np.float32).tofile("geoB_pos.bin"); body.vertex_normals.astype(np.float32).tofile("geoB_nrm.bin")
J=pickle.load(open("../zoom3/joints.pkl","rb")); eyes=[]
for s,k in [("l",0),("r",1)]:
    ev=M[f'ev{k}']; ef=M[f'ef{k}']; c=ev.mean(0); t=J[f'{s}-eye-target']-J[f'{s}-eye']; t/=np.linalg.norm(t)
    ca=((ev-c)/np.linalg.norm(ev-c,axis=1,keepdims=True))@t
    col=np.where(ca[:,None]>0.93,[0.10,0.07,0.05],np.where(ca[:,None]>0.80,[0.30,0.22,0.15],[0.93,0.91,0.89]))
    eyes.append(dict(v=ev.round(3).tolist(),f=ef.tolist(),c=col.round(3).tolist()))
S=json.load(open("spots.json"))
S["faceB"]=dict(S["face"],pos=pos.round(3).tolist(),nrm=res[reg]['nrm'].round(3).tolist()); S["_eyes"]=eyes
json.dump(S,open("spots.json","w")); print("ok", len(ix), "sigma %.2f"%np.median(nn))
