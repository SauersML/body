from symdraw import *
from multiprocessing import Pool
HOLD,MORPH,END=12,21,60
seq=[]
for i,L_ in enumerate(LAYERS):
    for _ in range(HOLD if i<len(LAYERS)-1 else END): seq.append((L_,L_,0.0))
    if i<len(LAYERS)-1:
        for f in range(1,MORPH+1): t=f/(MORPH+1); seq.append((L_,LAYERS[i+1],t*t*(3-2*t)))
def frame(k):
    a,b,t=seq[k]; V=(1-t)*KEY[a]+t*KEY[b]; lab=a if t<0.5 else b
    fig=plt.figure(figsize=(19.2,10.8),dpi=100); ax=fig.add_axes([0.0,0.0,1,1])
    draw(ax,V,lw=8,dot=60); ax.set_aspect('equal'); ax.set_ylim(2,228); ax.set_xlim(-205,205)
    fig.text(0.04,0.91,f'layer {lab}',fontsize=50,color=INK)
    fig.savefig(f'frames/f{k:04d}.png',facecolor='white'); plt.close(fig)
if __name__=='__main__':
    with Pool(12) as p: p.map(frame,range(len(seq)))
    print(len(seq))
