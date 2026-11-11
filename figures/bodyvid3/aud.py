import numpy as np, json, sys
sys.path.insert(0,'../bodyvid'); from audio import synth
FPS=60
def sm(x): x=np.clip(x,0,1); return x*x*x*(x*(6*x-15)+10)
S=json.load(open('web/spots.json')); XM=np.array(S['XM']); Yu=np.array(S['Yu']); lag=np.array(S['lag'])
T0,T1,DUR=5.0,34.0,8.0
def amt(t): return np.zeros(len(lag)) if t<T0 else (sm((t-T0-lag)/DUR) if t<T1 else 1-sm((t-T1-lag)/DUR))
d=np.linalg.norm(Yu-XM,axis=1)
c=[(d*amt(k/FPS)).mean() for k in range(int(52*FPS))]
synth(np.array(c),FPS,'c1.wav')
