from rend import *
from PIL import Image, ImageDraw
d = rest(); P, N = H.taxel_world(d)
c = Cam(900, 1200, [0, 0, 1.0], 2.9, 180, -5)
tips=[d.site("tip_left").xpos.copy(), d.site("tip_right").xpos.copy()]
im = c.render(d, dots=[(P + N * 0.003, [0.1, 0.1, 0.1, 1], 0.006),(tips,[1,0,0,1],0.015)])
xy = c.proj(tips); print(xy)
im=Image.fromarray(im, "RGBA"); bg=Image.new('RGBA',im.size,'white'); bg.alpha_composite(im); dr=ImageDraw.Draw(bg)
for x,y in xy: dr.ellipse([x-8,y-8,x+8,y+8],outline='blue',width=3)
bg.convert('RGB').save('t2.png')
