import pickle, numpy as np
from rend import *
S=pickle.load(open('steer_dump_Qwen3-8B_named.pkl','rb')); O=S['steer']
sim=E.Sim("mount", np.random.default_rng(0)); q0=sim.d.qpos.copy()
def tips(qs):
    out=[]
    for q in [q0]+list(qs):
        setpose(q); out.append([D.site("tip_left").xpos.copy(), D.site("tip_right").xpos.copy()])
    return np.array(out)  # [21,2,3]
R={}
for t in O:
    R[t]={}
    for c in O[t]:
        r=O[t][c][0]; hand=min(r['dist_hand'],key=r['dist_hand'].get)
        R[t][c]=dict(path=tips(r['qpos'])[:, 0 if hand=='left' else 1], hand=hand, dist=r['dist'], qend=r['qpos'][-1], touch=r['touch'])
pickle.dump(dict(R=R,q0=q0),open('paths.pkl','wb'))
for t in R: print(t, {c:(R[t][c]['hand'][0], round(100*R[t][c]['dist'],1)) for c in R[t]})
