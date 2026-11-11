from data import *
Z=model('ZRu'); U,s,Vt=np.linalg.svd(Z,full_matrices=False); PC=U*s
F=np.c_[np.abs(XYZ[:,0]),XYZ[:,1],XYZ[:,2]]; F=F-F.mean(0)   # folded body (cm)
def procrustes(A,B):  # map A (n,k) to B (n,3): orthonormal R (k,3), scale
    M=A.T@B; u,_,vt=np.linalg.svd(M,full_matrices=False); R=u@vt
    Y=A@R; c=(Y*B).sum()/(Y*Y).sum(); return R,c
def fold3(k=3):
    R,c=procrustes(PC[:,:k],F); return PC[:,:k]@R*c
