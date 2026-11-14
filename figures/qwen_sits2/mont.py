import sys
from PIL import Image
box=tuple(map(int,sys.argv[1].split(','))); fs=sys.argv[2:]
ims=[Image.open(f).convert('RGBA').crop(box) for f in fs]
m=Image.new('RGBA',(sum(i.width for i in ims),ims[0].height),'white'); x=0
for i in ims: m.alpha_composite(i,(x,0)); x+=i.width
m=m.convert('RGB'); m.thumbnail((1800,1000)); m.save('mont.png')
