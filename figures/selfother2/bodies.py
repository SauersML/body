import os
import json,numpy as np,sys
from anchors import KEYS, colorfield
from bodyrender import render
D=json.load(open(os.path.expandvars('$BODY_REPO/selfother/out3/figdata_plain.json')))['dp']
T=D['types']
def tk(k): return k[:-1] if k[-1] in "LR" and k not in ("neckF","neckB") else k
def cmap(t):
    stops=np.array([[0.86,0.85,0.84],[0.99,0.82,0.45],[0.95,0.45,0.12],[0.66,0.06,0.10]]); x=np.clip(t,0,0.999)*3; i=x.astype(int); f=(x-i)[:,None]; return stops[i]*(1-f)+stops[i+1]*f
VMAX=2.0
for nm in ['own','normal','me','random']:
    p=np.array(D['prof'][nm]); vals={k:p[T.index(tk(k))] for k in KEYS}
    f=colorfield(vals); render(cmap(np.clip(f,0,None)/VMAX),f'body_{nm}.png',views=(0,np.pi),W=700,Hh=1200)
