exec(open('drive.py').read()); exec(open('cams.py').read())
V=[]; x=0
for reg in CAM:
    for c in CAM[reg]:
        for src in "mw": V.append(dict(reg=reg,src=src,vp=[(len(V)%4)*600,(len(V)//4)*600,600,600],dots=0.15,**c))
shoot(dict(key=2.4,hemi=0.22,env=0.3,W=2400,H=1200*1+600*2,dpr=1,views=V,html=""),"test.png")
