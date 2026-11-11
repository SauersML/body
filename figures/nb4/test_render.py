exec(open('drive.py').read())
yc=89.5; hh=91
def V(col,front,vp): 
    s=1 if front else -1
    return dict(col=col,vp=vp,halfh=hh,eye=[0,yc,400*s],tgt=[0,yc,0],light=[-120*s,300,300*s])
cfg=dict(W=2400,H=1000,dpr=2,views=[V("32B",1,[0,0,600,1000]),V("32B",0,[600,0,600,1000]),V("8B",1,[1200,0,600,1000]),V("8B",0,[1800,0,600,1000])])
shoot(cfg,"test.png")
