import sys, numpy as np
from scipy.linalg import orthogonal_procrustes
K = int(sys.argv[2]) if len(sys.argv)>2 else 12
A = np.load(sys.argv[1], mmap_mode='r'); NL = A.shape[0]-1
Y = {}
for l in range(1, NL+1):
    Z = np.asarray(A[l], np.float64); Z = Z - Z.mean(0)
    u, s, vt = np.linalg.svd(Z, full_matrices=False); y = u[:, :K]*s[:K]; Y[l] = y/np.sqrt((y**2).sum())
AL = {NL: Y[NL]}
for l in range(NL-1, 0, -1): R_, _ = orthogonal_procrustes(Y[l], AL[l+1]); AL[l] = Y[l] @ R_
np.save(f'AL{K}.npy', np.stack([AL[l] for l in range(1, NL+1)]))
