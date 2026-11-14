from rend import *
from PIL import Image
d = rest(); P, N = H.taxel_world(d)
c = Cam(900, 1200, [0, 0, 1.0], 2.9, 180, -5)
im = c.render(d, dots=[(P + N * 0.003, [0.1, 0.1, 0.1, 1], 0.006)])
xy = c.proj([d.site("tip_left").xpos, d.site("tip_right").xpos, d.geom_xpos[M.geom("head").id]]); print(xy)
Image.fromarray(im, "RGBA").save("t1.png"); print(im[..., 3].mean())
