import pickle, numpy as np, sys
from rend import *
from PIL import Image, ImageDraw
P_=pickle.load(open('paths.pkl','rb')); R=P_['R']
M.light_castshadow[:] = 0
AZ=float(sys.argv[1]) if len(sys.argv)>1 else 165
c=Cam(700,1000,[0,0,0.95],2.75,AZ,-4)
def tile(t, cond, fn):
    r=R[t][cond]; setpose(r['qend']); Pw,Nw=H.taxel_world(D)
    if t=='none': tm=np.zeros(H.NT,bool)
    else: tm=H.TPART==H.PARTS.index(t)
    im=c.render(D,dots=[(Pw[~tm]+Nw[~tm]*0.003,[0.12,0.12,0.12,1],0.0045),(Pw[tm]+Nw[tm]*0.004,[0.82,0.26,0.18,1],0.009)])
    xy=c.proj(r['path']); Image.fromarray(im,'RGBA').save(fn); return xy
if __name__=='__main__':
    sheet=Image.new('RGB',(700*5,1000*2),'white')
    T=list(R)+['none']
    for k,t in enumerate(T):
        xy=tile(t if t!='none' else 'head', 'a1.5' if t!='none' else 'none', f'tile_{k}.png')
        im=Image.open(f'tile_{k}.png'); bg=Image.new('RGBA',im.size,'white'); bg.alpha_composite(im); dr=ImageDraw.Draw(bg)
        dr.line([tuple(p) for p in xy],fill=(43,95,168),width=6); dr.text((20,20),t,fill='black')
        sheet.paste(bg.convert('RGB'),((k%5)*700,(k//5)*1000))
    sheet.resize((1750,1000)).save('sheet.png')
