import numpy as np
H='$BODY_WORK/fig/human/'
def load(obj=H+'neutral.obj'):
    V=[];F=[];g=None
    for l in open(obj):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('g '): g=l.split()[1]
        elif l.startswith('f ') and g=='body': F.append([int(t.split('/')[0])-1 for t in l.split()[1:]])
    V=np.array(V);F=np.array(F); used=np.unique(F); rm=-np.ones(len(V),int); rm[used]=np.arange(len(used)); P=V[used]; F=rm[F]
    tri=np.vstack([F[:,[0,1,2]],F[:,[0,2,3]]])
    return P,tri
