import os
import numpy as np
V=[];F=[];g=None
for l in open(os.path.expandvars('$BODY_REPO/mesh/neutral.obj')):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): g=l.split()[1]
    elif l.startswith('f ') and g=='body': F.append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V);F=np.array(F)
used=np.unique(F); rm=-np.ones(len(V),int); rm[used]=np.arange(len(used)); P=V[used]; F=rm[F]
np.save('meshP.npy',P); np.save('meshF.npy',F)
print(P.shape,F.shape,P.min(0),P.max(0))
