import os
exec(open('drive.py').read())
V=[]
def add(reg,src,vp,eye,tgt,up,ext,dots=0.09): V.append(dict(reg=reg,src=src,vp=vp,eye=eye,tgt=tgt,up=up,ext=ext,dots=dots))
for k,src in enumerate("mw"):
  o=k*800
  add("face",src,[0,o,400,450],[14,167,60],[0,161.5,4],[0,1,0],13)
  add("hand",src,[400,o,300,450],[86,140,60],[86.5,145,0],[1,0,0],15)
  add("hand",src,[700,o,300,450],[86,140,-60],[86.5,145,0],[1,0,0],15)
  add("foot",src,[1000,o,600,300],[-55,25,12],[13,7,7],[0,1,0],10)
  add("foot",src,[1000,o+300,600,300],[55,-25,12],[13,5,7],[0,1,0],10)
  add("forearm",src,[0,o+450,800,300],[62,165,60],[62,143,0],[0,1,0],10)
  add("forearm",src,[800,o+600,800,200],[62,121,-60],[62,143,0],[0,-1,0],10)
shoot(dict(W=1600,H=1600,dpr=1,views=V),os.path.expandvars("$BODY_REPO/figures/zoom2/test.png"))
