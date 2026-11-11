import os
import numpy as np, json
S=json.load(open(os.path.expandvars("$BODY_REPO/figures/zoom3/spots.json"))); Fr={k:np.array(v) for k,v in S["_frames"].items()}
def nz(v): v=np.asarray(v,float); return v/np.linalg.norm(v)
def cam(tgt,d,up,ext,fov=16,lx=-0.5,ly=0.7,lz=0.6,shift=(0,0)):
    d=nz(d); up=nz(np.asarray(up,float)-d*(np.dot(up,d))); r=np.cross(-d,up)
    tgt=np.asarray(tgt,float)+shift[0]*r+shift[1]*up
    dist=ext/np.tan(np.radians(fov/2)); eye=tgt+d*dist
    light=tgt+120*nz(lz*d+lx*r+ly*up)
    return dict(eye=eye.tolist(),tgt=tgt.tolist(),up=up.tolist(),fov=fov,light=light.tolist(),sext=ext*1.8)
def C(reg): return np.array(S[reg]["pos"]).mean(0)
ez,ex,ey=Fr["ez"],Fr["ex"],Fr["ey"]; W,E=Fr["W"],Fr["E"]; ax=nz(W-E); Rf=Fr["Rf"]; fz=Rf[:,2]
FA=0.5*np.linalg.norm(W-E)*1.45/(840/330)
CAM={
 "face":[cam(C("face")+[0,1.6,0],[0.28,0.06,1],[0,1,0],14)],
 "hand":[dict(cam(C("hand"),ez,ex,15.0,shift=(0,-3.2)),clip=[1,0,0,30]), dict(cam(C("hand"),-ez,ex,15.0,shift=(0,-3.2)),clip=[1,0,0,30])],
 "foot":[dict(cam(C("foot")+[0,4.5,0],[-1,0.32,0.15],[0,1,0],10.5,lx=0.3),clip=[1,0,0,8]), dict(cam(C("foot")+[0,4.5,0],[1,0.32,0.15],[0,1,0],10.5,lx=-0.3),clip=[1,0,0,8])],
 "forearm":[dict(cam((E+W)/2,fz,np.cross(ax,-fz),FA),clip=[1,0,0,26]), dict(cam((E+W)/2,-fz,np.cross(ax,fz),FA),clip=[1,0,0,26])],
}
