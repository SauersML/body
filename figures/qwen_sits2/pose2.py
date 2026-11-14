import numpy as np, json, sys, scipy.sparse as sp
from scipy.spatial.transform import Rotation as Rot
J=np.load('joints.npy',allow_pickle=True).item()
A=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
g=lambda k,d: A.get(k,d)
V=[];F=[];grp=None
for l in open('../human/neutral.obj'):
    if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith('g '): grp=l.split()[1]
    elif l.startswith('f ') and grp=='body': F.append([int(t.split('/')[0])-1 for t in l.split()[1:]])
V=np.array(V);F=np.array(F); used=np.unique(F); rm=-np.ones(len(V),int); rm[used]=np.arange(len(used)); P=V[used]; F=rm[F]
nz=lambda v: v/np.linalg.norm(v)
def align(a,b):
    a,b=nz(a),nz(b); v=np.cross(a,b); c=a@b
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+K+K@K/(1+c)
def frame(a,n):  # orthonormal frame from axis a and approx normal n
    a=nz(a); n=nz(n-(n@a)*a); return np.stack([a,n,np.cross(a,n)],1)
def frot(a0,n0,a1,n1): return frame(a1,n1)@frame(a0,n0).T
def rx(t): return Rot.from_euler('x',t,degrees=True).as_matrix()
bones={}   # name:(a,b,side,R,jr,jn)
pel=J['pelvis']; lean=rx(g('lean',0.0))
bones['torso']=(J[A.get('tstart','spine-4')],J['head'],0,lean,pel,pel)
T=lambda name: (lambda b:(b[3],b[4],b[5]))(bones[name])
def apply(name,x): R,jr,jn=T(name); return R@(x-jr)+jn
for s,sg in [('l',1),('r',-1)]:
    hip,kn,an,ft=J[s+'-upper-leg'],J[s+'-knee'],J[s+'-ankle'],J[s+'-foot-2']
    Rt=align(kn-hip, np.array([g('kx',0.10)*sg,g('ky',-0.04),1.0])); k2=hip+Rt@(kn-hip)
    Rs=align(an-kn, np.array([g('sx',0.03)*sg,-1,g('sz',0.06)])); a2=k2+Rs@(an-kn)
    bones[s+'thigh']=(hip,kn,sg,Rt,hip,hip); bones[s+'shin']=(kn,an,sg,Rs,kn,k2)
    bones[s+'foot']=(an,ft,sg,np.eye(3),an,a2)
    sh,el,ha,fi=J[s+'-shoulder'],J[s+'-elbow'],J[s+'-hand'],J[s+'-finger-3-1']
    th,pk=J[s+'-finger-1-1'],J[s+'-finger-5-1']
    sh2=apply('torso',sh)+np.array([-g('sdx',0.0)*sg,-g('sd',0.0),-g('sdz',0.0)])
    Lu=np.linalg.norm(el-sh); Lf=np.linalg.norm(ha-el)
    tgt=np.array([g('hx',0.9)*sg,g('hy',1.55),g('hz',2.3)])
    dv=tgt-sh2; dd=min(np.linalg.norm(dv),Lu+Lf-1e-3)
    a=(Lu**2-Lf**2+dd**2)/(2*dd); hgt=np.sqrt(max(Lu**2-a**2,0)); u=nz(dv)
    pole=np.array([g('px',0.35)*sg,0,-1.0]); pole=nz(pole-(pole@u)*u); e2=sh2+a*u+hgt*pole
    Ru=align(el-sh,e2-sh2); bones[s+'uarm']=(sh+g('us',0.0)*(el-sh),el,sg,Ru,sh,sh2)
    # palm normal at rest: points medially for an A-pose hand
    n0=nz(np.cross(fi-ha, th-pk)); n0=n0 if n0[0]*sg<0 else -n0
    h2=e2+Lf*nz(tgt-e2)
    nt=np.array([g('nx',-0.25)*sg,-1.0,g('nz',0.0)])            # palm faces down onto the thighs
    dt=nz(np.array([-g('dx',0.55)*sg,g('dy',-0.25),1.0]))        # fingers forward and inward
    Rh=frot(fi-ha,n0,dt,nt)
    Rf=frot(ha-el,n0,h2-e2,Rh@n0)   # forearm carries the pronation
    bones[s+'farm']=(el,ha,sg,Rf,el,e2)
    bones[s+'hand']=(ha,fi+(fi-ha)*1.2,sg,Rh,ha,h2)
names=list(bones)
def segd(p,a,b):
    ab=b-a; t=np.clip(((p-a)@ab)/(ab@ab),0,1); return np.linalg.norm(p-(a+t[:,None]*ab),axis=1)
D=np.stack([segd(P,bones[n][0],bones[n][1]) for n in names],1)
for i,n in enumerate(names):
    if bones[n][2]: D[P[:,0]*bones[n][2]<0.15,i]=1e9
W=1/np.maximum(D,0.05)**8; W/=W.sum(1,keepdims=True)
e=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,3]],F[:,[3,0]]]); Ad=sp.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(P),)*2); Ad=((Ad+Ad.T)>0).astype(float)
dg=np.asarray(Ad.sum(1)).ravel()
for _ in range(g('smooth',40)): W=0.5*W+0.5*(Ad@W)/dg[:,None]
# dual-quaternion skinning
def qmul(a,b):
    w1,x1,y1,z1=a.T; w2,x2,y2,z2=b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],-1)
Q0=[];Qe=[]
for n in names:
    a,b,sg,R,jr,jn=bones[n]; t=jn-R@jr
    q=Rot.from_matrix(R).as_quat(); q=np.array([q[3],q[0],q[1],q[2]])
    if q[0]<0: q=-q
    Q0.append(q); Qe.append(0.5*qmul(np.array([0,*t]),q))
Q0=np.array(Q0);Qe=np.array(Qe)
b0=W@Q0; be=W@Qe; nrm=np.linalg.norm(b0,axis=1,keepdims=True); b0/=nrm; be/=nrm
w=b0[:,0:1]; v=b0[:,1:]; 
Pr=P+2*np.cross(v,np.cross(v,P)+w*P)
t=2*(qmul(be,b0*np.array([1,-1,-1,-1]))[:,1:])
Q=Pr+t
np.save('posed.npy',dict(P=Q,F=F),allow_pickle=True)
th=(W[:,names.index('lthigh')]+W[:,names.index('rthigh')])>0.6
seat=float(Q[th,1].min()); back=float(Q[(W[:,0]>0.6)&(Q[:,1]>seat+1),2].min()); floor=float(Q[:,1].min())
json.dump(dict(seat=seat,back=back,floor=floor,hipw=float(np.abs(Q[th,0]).max())),open('chair.json','w')); print(open('chair.json').read())
