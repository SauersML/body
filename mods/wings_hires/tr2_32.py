import sys, numpy as np, json
sys.argv=['x','out32b','0']; import ana as A
O='out32b'
def T(J, a, b):
    Xa=np.load(f"{O}/H_{a}.npy")[:,J].astype(np.float32); Xb=np.load(f"{O}/H_{b}.npy")[:,J].astype(np.float32)
    return A.transfer(Xa,Xb)[0]
for J,L in enumerate([32,40,48,52,56]):
    for a,b in [("W_wing","W_desc"),("N_wing","N_desc"),("G_glorp","G_desc"),("F_wing","N_desc")]:
        r=T(J,a,b); print(L,f"{a}>{b}"," ".join(f"{k}={v:.2f}" for k,v in r.items() if k!='pos_pred'))
