import numpy as np, json
J=np.load('joints.npy',allow_pickle=True).item()
V=[];F=[];g=None
for l in open('../human/neutral.obj'):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): g=l.split()[1]
    elif l.startswith('f ') and g=='body': F.append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V);F=np.array(F); used=np.unique(F); rm=-np.ones(len(V),int); rm[used]=np.arange(len(used)); P=V[used]; F=rm[F]
def nz(v): return v/np.linalg.norm(v)
def align(a,b):
    a,b=nz(a),nz(b); v=np.cross(a,b); c=a@b
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+K+K@K/(1+c)
def rx(t):
    t=np.radians(t); c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
bones={}  # name: (a,b, side, R, j_rest, j_new)
lean=rx(-2)   # torso leans back a little
pel=J['pelvis']
bones['torso']=(J['spine-4'],J['head'],0,lean,pel,pel)
for s,sg in [('l',1),('r',-1)]:
    hip,kn,an,ft=J[s+'-upper-leg'],J[s+'-knee'],J[s+'-ankle'],J[s+'-foot-2']
    Rt=align(kn-hip, np.array([0.18*sg,-0.02,1.0]))
    k2=hip+Rt@(kn-hip)
    Rs=align(an-kn, np.array([0.12*sg,-1,0.12]))
    a2=k2+Rs@(an-kn)
    Rf=np.eye(3)
    bones[s+'thigh']=(hip,kn,sg,Rt,hip,hip)
    bones[s+'shin']=(kn,an,sg,Rs,kn,k2)
    bones[s+'foot']=(an,ft,sg,Rf,an,a2)
    sh,el,ha,fi=J[s+'-shoulder'],J[s+'-elbow'],J[s+'-hand'],J[s+'-finger-3-1']
    sh2=pel+lean@(sh-pel)
    Lu=np.linalg.norm(el-sh); Lf0=np.linalg.norm(ha-el)
    T=np.array([1.45*sg,1.75,2.7]); dv=T-sh2; dd=min(np.linalg.norm(dv),Lu+Lf0-1e-3)
    a=(Lu**2-Lf0**2+dd**2)/(2*dd); hgt=np.sqrt(max(Lu**2-a**2,0)); u=nz(dv)
    pole=np.array([0.5*sg,0,-1.0]); pole=nz(pole-(pole@u)*u)
    e2=sh2+a*u+hgt*pole
    Ru=align(el-sh,e2-sh2)
    Lf=np.linalg.norm(ha-el); 
    bones[s+'uarm']=(sh,el,sg,Ru,sh,sh2)
    bones[s+'_tmp']=(e2,Lf)
order=[k for k in bones if not k.endswith('_tmp')]
# thigh top for hand target
for s,sg in [('l',1),('r',-1)]:
    e2,Lf=bones.pop(s+'_tmp')
    el,ha,fi=J[s+'-elbow'],J[s+'-hand'],J[s+'-finger-3-1']
    hip=J[s+'-upper-leg']; Rt=bones[s+'thigh'][3]; kn2=hip+Rt@(J[s+'-knee']-hip)
    tgt=np.array([1.45*sg,1.75,2.7])
    Rf=align(ha-el,tgt-e2)
    # rotate the forearm also so the palm faces down: extra twist not applied
    h2=e2+Rf@(ha-el)
    bones[s+'farm']=(el,ha,sg,Rf,el,e2)
    bones[s+'hand']=(ha,fi+(fi-ha)*1.2,sg,Rf,ha,h2)
names=list(bones)
def segd(p,a,b):
    ab=b-a; t=np.clip(((p-a)@ab)/(ab@ab),0,1); return np.linalg.norm(p-(a+t[:,None]*ab),axis=1)
D=np.stack([segd(P,bones[n][0],bones[n][1]) for n in names],1)
for i,n in enumerate(names):
    sg=bones[n][2]
    if sg: D[P[:,0]*sg<0.15,i]=1e9
    if n=='torso': D[:,i]*=1.0
W=1/np.maximum(D,0.05)**8; W/=W.sum(1,keepdims=True)
# smooth weights over mesh adjacency a few times
import scipy.sparse as sp
e=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,3]],F[:,[3,0]]]); A=sp.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(P),)*2); A=((A+A.T)>0).astype(float)
dg=np.asarray(A.sum(1)).ravel(); 
for _ in range(40): W=0.5*W+0.5*(A@W)/dg[:,None]
Q=np.zeros_like(P)
for i,n in enumerate(names):
    a,b,sg,R,jr,jn=bones[n]; Q+=W[:,i:i+1]*((P-jr)@R.T+jn)
np.save('posed.npy',dict(P=Q,F=F),allow_pickle=True)
print('min y',Q[:,1].min(),'seat est', [Q[(W[:,names.index(s+'thigh')]>0.6),1].min() for s in 'lr'], 'zmin',Q[:,2].min(),'zmax',Q[:,2].max(),'x',Q[:,0].min(),Q[:,0].max())
# seat: lowest y of thigh-dominant verts; back: min z of torso verts above seat
th=(W[:,names.index('lthigh')]+W[:,names.index('rthigh')])>0.6
seat=float(Q[th,1].min()); back=float(Q[(W[:,0]>0.6)&(Q[:,1]>seat+1),2].min()); floor=float(Q[:,1].min())
kn=max(bones['lshin'][5][2],0)
json.dump(dict(seat=seat,back=back,floor=floor,knee_z=float(kn),hipw=float(np.abs(Q[th,0]).max())),open('chair.json','w')); print(open('chair.json').read())
