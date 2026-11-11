"""Map each MakeHuman vertex to the stimulus T-pose frame (bone-local), so spot values can be painted on the mesh."""
import os
import numpy as np, pickle, sys
sys.path.insert(0,os.path.expandvars('$BODY_REPO/mods/base')); import stim as S, mstim as MS
M=np.load(os.path.expandvars('$BODY_REPO/mods/phantom_fig_v3/mesh/mesh.npz')); V=M['v']; F=M['f']
J=pickle.load(open(os.path.expandvars('$BODY_REPO/mods/phantom_fig_v3/mesh/joints.pkl'),'rb'))
B=[]  # (mesh a, mesh b, stim a, stim b, radius, group[, frame])
FR={}
for s,c in ((1,'l'),(-1,'r')):
    side='L' if s>0 else 'R'
    j=lambda n: J[f'{c}-{n}']
    nz=lambda v: v/np.linalg.norm(v)
    ex=nz(j('finger-3-1')-j('hand')); sd=j('finger-2-1')-j('finger-5-1'); ey=nz(sd-ex*(sd@ex)); ez=np.cross(ex,ey)
    Ms=np.stack([ex,ey,ez],1); exs=np.array([s,0,0.]); eys=np.array([0,1,0.]); Ss=np.stack([exs,eys,np.cross(exs,eys)],1)
    FR['hand'+side]=Ss@Ms.T
    ax=nz(j('hand')-j('elbow')); fy=nz(ey-ax*(ey@ax)); Mf=np.stack([ax,fy,np.cross(ax,fy)],1); FR['fore'+side]=Ss@Mf.T
    B+= [(j('shoulder'),j('elbow'),(s*20,143,0),(s*50,143,0),4.5,'arm'+side),
         (j('elbow'),j('hand'),(s*50,143,0),(s*76,143,0),3.8,'arm'+side,'fore'+side),
         (j('hand'),j('finger-3-1'),(s*76,143,0),(s*91,142.5,0),2.2,'hand'+side,'hand'+side)]
    MC={2:(146,99),3:(142.5,100),4:(139,99.5),5:(135.5,97.5)}
    for k,(y,xt) in MC.items():
        a=j(f'finger-{k}-1'); b=j(f'finger-{k}-4'); b=b+0.35*(b-j(f'finger-{k}-3'))
        B.append((a,b,(s*91,y,0),(s*xt,y,0),0.8,'hand'+side,'hand'+side))
    a=j('finger-1-1'); b=j('finger-1-4'); b=b+0.35*(b-j('finger-1-3'))
    B.append((a,b,(s*79,145,0),(s*87,157.5,0),0.9,'hand'+side,'hand'+side))
    B+= [(j('upper-leg'),j('knee'),(s*10,92,0),(s*12,50,0),7,'leg'+side),
         (j('knee'),j('ankle'),(s*12,50,0),(s*13,9,0),5,'leg'+side),
         (j('ankle'),j('toe-3-1'),(s*13,6,-2),(s*13,1,17),3,'foot'+side),
         (j('clavicle'),j('shoulder'),(s*4,146,0),(s*20,145,0),5,'torso')]
B+= [(J['pelvis']+[0,-5,0],J['neck'],(0,88,0),(0,150,0),12,'torso'),(J['neck'],J['head-2'] if J['head-2'][1]>J['head'][1] else J['head']+[0,14,0],(0,150,0),(0,176,0),8.5,'head')]
print('head-2',J['head-2'],'head',J['head'])
def rot(u,v):
    u=u/np.linalg.norm(u); v=v/np.linalg.norm(v); w=np.cross(u,v); c=u@v; K=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    return np.eye(3)+K+K@K/(1+c)
best=np.full(len(V),np.inf); P=np.zeros((len(V),3)); G=np.empty(len(V),object)
for bb in B:
    ma,mb,sa,sb,r,g=bb[:6]; fr=bb[6] if len(bb)>6 else None
    ma,mb,sa,sb=map(lambda x:np.array(x,float),(ma,mb,sa,sb))
    d=mb-ma; t=np.clip(((V-ma)@d)/(d@d),0,1); foot=ma+t[:,None]*d; off=V-foot; dist=np.linalg.norm(off,axis=1)-r
    R=FR[fr] if fr else rot(d,sb-sa); ps=sa+t[:,None]*(sb-sa)+off@R.T
    m=dist<best; best[m]=dist[m]; P[m]=ps[m]; G[m]=g
np.savez('vmap.npz',P=P,G=G.astype(str))
import collections; print(collections.Counter(G))
