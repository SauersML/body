import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P=np.load('meshP.npy')
def v(name, sel, a, b, lim=None, step=0.5):
    Q=P[sel]; fig,ax=plt.subplots(figsize=(10,10)); ax.scatter(Q[:,a],Q[:,b],s=1,c=Q[:,3-a-b],cmap='viridis'); ax.set_aspect('equal')
    ax.grid(True); ax.set_xticks(np.arange(np.floor(Q[:,a].min()),Q[:,a].max()+step,step)); ax.set_yticks(np.arange(np.floor(Q[:,b].min()),Q[:,b].max()+step,step))
    ax.tick_params(labelsize=7); plt.xticks(rotation=90); fig.savefig(name,dpi=90,bbox_inches='tight'); plt.close()
x,y,z=P.T
v('lk_front.png', x>-0.01, 0,1, step=1)
v('lk_hand.png', (x>3)&(y<2.5), 0,1, step=0.2)
v('lk_foot_side.png', (x>0)&(y<-6), 2,1, step=0.2)
v('lk_head_side.png', (y>5), 2,1, step=0.2)
v('lk_arm_top.png', (x>1.5)&(y>0.8), 0,2, step=0.25)
v('lk_arm_side.png', (x>2.5)&(y>0.8), 2,1, step=0.25)
