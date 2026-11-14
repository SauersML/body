import numpy as np, pickle
from colors import pcs, gauss, to_rgb2
from render import render
from PIL import Image, ImageDraw, ImageFont
r=np.load('$BODY_RUNS/reps_Qwen3-8B_named.npz')
HF,ti=r['HF'],r['ti']; RM=np.load('robot_mesh.npz'); TW,TG=RM['TW'],RM['TG']; meshes=pickle.load(open('robot_mesh.pkl','rb'))
V=np.vstack([v for _,v,_ in meshes]); off=np.cumsum([0]+[len(v) for _,v,_ in meshes])
F=np.vstack([f+off[k] for k,(_,_,f) in enumerate(meshes)]); G=np.concatenate([[g]*len(v) for g,v,_ in meshes])
def colours(Z):
    Y,_=pcs(Z,3); Y=gauss(Y)
    U=np.unique(ti); Yt=np.array([Y[ti==u].mean(0) for u in U]); Pt=TW[U]; Gt=TG[U]
    out=np.zeros((len(V),3))
    for g in np.unique(G):
        m=G==g; ok=Gt==g
        if ok.sum()<1: ok=np.ones(len(U),bool)
        D=np.linalg.norm(V[m][:,None]-Pt[ok][None],axis=-1); w=np.exp(-(D-D.min(1,keepdims=True))**2/(2*0.03**2)); w/=w.sum(1,keepdims=True)
        out[m]=w@Yt[ok]
    return to_rgb2(out,np.radians(120),Lc=0.64,Ls=0.17,C=0.14,mode='raw',sq=1.0)
cq=colours(HF); cr=colours(np.random.default_rng(0).standard_normal(HF.shape))
PP=np.c_[V[:,1],V[:,2],V[:,0]]
PW,PH,TOP=1000,1500,150; W=4*PW; H=TOP+PH
pan=[dict(col=c,ang=a,vp=(i*PW,0,PW,PH),hh=0.86,yc=0.80) for i,(c,a) in enumerate([(cq,0),(cq,np.pi),(cr,0),(cr,np.pi)])]
render(pan,'f3_raw.png',W,H,bg='#ffffff',PP=PP,TT=F)
im=Image.open('f3_raw.png').convert('RGB'); d=ImageDraw.Draw(im); FN=ImageFont.truetype('/System/Library/Fonts/HelveticaNeue.ttc',84)
for i,t in enumerate(['Qwen3-8B controller','random vectors']):
    w=d.textlength(t,font=FN); d.text((i*2*PW+PW-w/2,40),t,font=FN,fill='#1D1D1F')
im.save('../bodymap_art_3.png'); im.save('../bodymap_art_3.jpg',quality=93); im.resize((W//3,H//3)).save('f3_s.png')
