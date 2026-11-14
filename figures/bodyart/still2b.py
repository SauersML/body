from symdraw import *
from PIL import Image
fig=plt.figure(figsize=(26,24),dpi=100)
ax=fig.add_axes([0.02,0.02,0.96,0.86]); draw(ax,KEY[53],lw=13,dot=140)
V=KEY[53]; m=np.abs(V[:,0]).max()+8; ax.set_xlim(-m,m); ax.set_ylim(V[:,1].min()-8,V[:,1].max()+8)
fig.text(0.5,0.945,'the body inside Qwen3-32B',fontsize=72,color=INK,ha='center')
fig.text(0.5,0.905,'right-side spots drawn mirrored',fontsize=52,color=INK,ha='center')
fig.savefig('$BODY_WORK/fig/bodymap_art_2b.png',facecolor='white')
Image.open('$BODY_WORK/fig/bodymap_art_2b.png').convert('RGB').save('$BODY_WORK/fig/bodymap_art_2b.jpg',quality=93)
