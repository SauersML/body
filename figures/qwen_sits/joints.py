import numpy as np
V=[];g=None;G={}
for l in open('../human/base.obj'):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): g=l.split()[1]
    elif l.startswith('f ') and g.startswith('joint-'):
        G.setdefault(g,set()).update(int(t.split('/')[0])-1 for t in l.split()[1:])
V=np.array(V)
J={k[6:]:V[list(s)].mean(0) for k,s in G.items()}
np.save('joints.npy',J,allow_pickle=True)
for k in ['pelvis','spine-1','spine-4','neck','head','l-upper-leg','l-knee','l-ankle','l-foot-1','l-foot-2','l-clavicle','l-shoulder','l-elbow','l-hand','l-finger-3-1','r-upper-leg','ground']: print(k,J[k].round(2))
