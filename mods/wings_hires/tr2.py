import sys, numpy as np, json
sys.argv=['x','out8b','0']; import ana as A
O='out8b'
def T(J, a, b):
    Xa=np.load(f"{O}/H_{a}.npy")[:,J].astype(np.float32); Xb=np.load(f"{O}/H_{b}.npy")[:,J].astype(np.float32)
    return A.transfer(Xa,Xb)[0]
for J,L in enumerate([12,18,24,27,32]):
    for a,b in [("W_wing","W_desc"),("N_wing","N_desc"),("G_glorp","G_desc"),("F_wing","N_desc")]:
        r=T(J,a,b); print(L,f"{a}>{b}"," ".join(f"{k}={v:.2f}" for k,v in r.items() if k!='pos_pred'))
