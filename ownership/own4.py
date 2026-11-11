# specificity: is the left-right effect on "mine" due to LR's overlap with the owner axis?
import json, numpy as np, torch
src=open("own3.py").read(); exec(src.split("CELLS=[")[0])
A=np.load("so_dirs.npz"); me=torch.tensor(A["d24"]).float().to(dev)
def orth(v,*us):
    for u in us: u=u/u.norm(); v=v-(v@u)*u
    return v/v.norm()
lrO=orth(lr,gY,gD); 
info={"cos(LR,you-woman)":float(lr@gY/gY.norm()),"cos(LR,you-doll)":float(lr@gD/gD.norm()),"cos(LR,selfother me)":float(lr@me/me.norm()),"cos(you-woman,me)":float(gY@me/gY.norm()/me.norm())}
log(info)
Zc=states([U(t) for p in [("Rain falls on your car.","Rain falls on her car."),("Someone scratches your car with a key.","Someone scratches her car with a key."),("A cat sits on your coat.","A cat sits on her coat."),("Rain falls on your phone.","Rain falls on her phone."),("Your bicycle is stolen.","Her bicycle is stolen."),("Your guitar falls over.","Her guitar falls over.")] for t in p])
car=torch.tensor((Zc[0::2,k]-Zc[1::2,k]).mean(0)).float().to(dev); info["cos(car,you-woman)"]=float(car@gY/car.norm()/gY.norm()); carO=orth(car,gY,gD)
out={"info":info}
for p,n in [("doll",nD),("woman",nY)]:
    for c,v in [("none",None),("left-right, owner part removed x2",lrO*2*n),("your-car dir x2",car/car.norm()*2*n),("your-car dir, owner part removed x2",carO*2*n),("selfother me x1",me),("selfother me x2",2*me)]+[(f"random{i} x2",rnd(2*n)) for i in range(6)]:
        HOOK.clear(); 
        if v is not None: HOOK[L0]=[add(v)]
        out[p+" | "+c]=fc(p); log(p,c,out[p+" | "+c])
HOOK.clear(); json.dump(out,open("res4.json","w")); log("DONE4")
