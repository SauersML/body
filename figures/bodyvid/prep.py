from geo import *
from scipy.spatial.distance import cdist
Fo=np.c_[np.abs(XYZ[:,0]),XYZ[:,1],XYZ[:,2]]; mu=Fo.mean(0)
R,c=procrustes(PC[:,:4],Fo-mu); Y=PC[:,:4]@R
Y=Y*np.sqrt(((Fo-mu)**2).sum()/(Y**2).sum())+mu        # same RMS size as the body; one uniform scale
# model's own left/right: shared shift direction, leave-pair-out
m=SIDE!=0; sd=np.zeros(407)
for i in range(407):
    if SIDE[i]==0: continue
    mm=m.copy(); mm[i]=False; mm[PART[i]]=False
    d=((Z[mm]-Z[PART[mm]])*SIDE[mm,None]).mean(0); sd[i]=np.sign((Z[i]-Z[PART[i]])@d)
print('decoded side acc',(sd[m]==SIDE[m]).mean())
DZ=cdist(Z,Z); np.fill_diagonal(DZ,np.inf); DF=cdist(Fo,Fo)
nn=np.argsort(DZ,1); err=DF[np.arange(407)[:,None],nn[:,:5]].mean(1)
E=sorted({(min(i,j),max(i,j)) for i in range(407) for j in nn[i,:2]})
np.savez('geo.npz',body=XYZ,Y=Y,sd=sd,err=err,E=np.array(E),side=SIDE,cat=CAT,part=PART,nrm=NRM)
print(len(E),'edges; spine u of midline',np.round(np.percentile(Y[SIDE==0,0],[10,50,90]),1), 'u range',Y[:,0].min(),Y[:,0].max())
