import numpy as np, sys, json
sys.path.insert(0,'../bodyvid'); from audio import synth
FPS=60
def sm(x): x=np.clip(x,0,1); return x*x*x*(x*(6*x-15)+10)
S=json.load(open('web/spots.json')); XM=np.array(S['XM']); Y=np.array(S['Y']); cat=S['cat']
order={'head':0,'arm':1,'hand':2,'torso':1.5,'leg':2.5,'foot':3}
lag=np.array([order[c] for c in cat])/3*1.2+(XM[:,1].max()-XM[:,1])/177*0.3
def m1(t): return np.zeros(len(lag)) if t<4.5 else (sm((t-4.5-lag)/5.5) if t<20 else 1-sm((t-20-lag)/5.5))
c1=[(np.linalg.norm(Y-XM,axis=1)*m1(k/FPS)).mean() for k in range(int(34*FPS))]
t=np.arange(int(40*FPS))/FPS; c2=sm((t-8)/8)+0.6*sm((t-18)/3.5)-1.6*sm((t-27)/4.5)
t=np.arange(int(32*FPS))/FPS; c3=np.where(t<20,sm((t-5)/7),1-sm((t-20)/7))
for v,c in (('v1',c1),('v2',c2),('v3',c3)): synth(np.array(c),FPS,v+'.wav')
