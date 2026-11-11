import numpy as np, json
from PIL import Image, ImageChops
def nb(im): im=im.convert('RGB'); return ImageChops.difference(im, Image.new('RGB',im.size,'white')).convert('L').point(lambda v:255 if v>6 else 0).getbbox()
from anchors import KEYS, A, P
from bodyrender import render
# plain gray body front, with anchor dot positions
cols=np.tile([[0.80,0.79,0.78]],(len(P),1)); W,Hh,yc,hh=700,1200,0.15,8.75
render(cols,'body_plain.png',views=(0,),W=W,Hh=Hh)
hw=hh*W/Hh
dots=[ [ (A[k][0]/hw*0.5+0.5)*W, (1-(A[k][1]-yc)/hh)*0.5*Hh, k] for k in KEYS if A[k][2]>-0.2]
im=Image.open('body_plain.png'); bb=nb(im); im.crop(bb).save('body_plain_c.png')
json.dump({"bbox":bb,"dots":[[x-bb[0],y-bb[1],k] for x,y,k in dots]},open('plain_dots.json','w'))
for n in ['own','normal','me','random']:
    im=Image.open(f'body_{n}.png'); b=nb(im); im.crop((b[0],b[1],b[2],b[3])).save(f'body_{n}_c.png'); print(n,b)
print(bb)
