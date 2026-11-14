import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from chains import *; from colors import *
R=np.load('$BODY_WORK/body/hires2/red32_short.npz')
sets={'ZR32':data()['q32'],'raw53':R['reply|53|all|A'].astype(np.float32),'ZR8':data()['q8']}
fig,ax=plt.subplots(3,3,figsize=(15,15))
for r,(k,Z) in enumerate(sets.items()):
    Y,_=pcs(Z,3)
    for c,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
        a=ax[r,c]; a.scatter(Y[:,i],Y[:,j],s=4,c='0.7')
        for side,col in [('left','C3'),('right','C1')]:
            for nm,ch in chains(side).items():
                P=np.array([Y[g].mean(0) for g in ch]); a.plot(P[:,i],P[:,j],'-',c=col if nm=='arm' else ('C2' if side=='left' else 'C9'))
        P=np.array([Y[g].mean(0) for g in spine()]); a.plot(P[:,i],P[:,j],'k-'); a.plot(P[0,i],P[0,j],'ko')
        a.set_title(f'{k} pc{i}{j}')
plt.savefig('draft2.png',dpi=60)
