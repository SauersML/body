import numpy as np, sys
from colors import *; from paint import paint,SP,SR; from render import render
from PIL import Image, ImageDraw, ImageFont
Dd=data(); Y32,_=pcs(Dd['q32']); Y8,_=pcs(Dd['q8']); Yr,_=pcs(Dd['rnd'])
for j in range(3): Y8[:,j]*=np.sign(np.corrcoef(Y8[:,j],Y32[:,j])[0,1])
s=np.sign(np.corrcoef(Y32[:,2],SP[:,1])[0,1]); Y32[:,2]*=s; Y8[:,2]*=s
TH=np.radians(float(sys.argv[1]) if len(sys.argv)>1 else 120)
cols=[paint(to_rgb2(Y,TH,Lc=0.64,Ls=0.17,C=0.14),sigma=3.0,smooth=10) for Y in (Y32,Y8,Yr)]
PW,PH,TOP=1100,1720,150; W=3*PW; H=TOP+2*PH
pan=[]
for i,c in enumerate(cols):
    for r,a in enumerate((0,np.pi)):
        pan.append(dict(col=c,ang=a,vp=(i*PW,(1-r)*PH,PW,PH),hh=8.55,yc=0.2))
render(pan,'f1_raw.png',W,H,bg='#ffffff')
im=Image.open('f1_raw.png').convert('RGB'); d=ImageDraw.Draw(im)
F=ImageFont.truetype('/System/Library/Fonts/HelveticaNeue.ttc',84,index=0)
for i,t in enumerate(['Qwen3-32B','Qwen3-8B','random vectors']):
    w=d.textlength(t,font=F); d.text((i*PW+PW/2-w/2,40),t,font=F,fill='#1D1D1F')
im.save('$BODY_WORK/fig/bodymap_art_1.png'); im.convert('RGB').save('$BODY_WORK/fig/bodymap_art_1.jpg',quality=93)
im.resize((W//3,H//3),Image.LANCZOS).save('f1_s.png')
