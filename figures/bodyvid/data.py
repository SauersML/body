import os
import sys, numpy as np, importlib.util
B=os.path.expandvars("$BODY_WORK/body/")
def mod(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
s1=mod('s1',os.path.expandvars('$BODY_REPO/hires/v1/stim.py')); s2=mod('s2',os.path.expandvars('$BODY_REPO/hires/v2/')+'stim.py')
L=s1.LOCS; P1=[l['phrase'] for l in L]; P2=[l['phrase'] for l in s2.LOCS]
IDX=np.array([P2.index(p) for p in P1])
XYZ=np.array([l['xyz'] for l in L],float); NRM=np.array([l['n'] for l in L],float)
SIDE=np.array([l['side'] for l in L]); REG=[l['region'] for l in L]
def cat(r): return "head" if r in ("head","neck") else "torso" if r=="torso" else r[:-1]
CAT=np.array([cat(r) for r in REG])
# mirror partner: swap left/right words
def sw(p): return p.replace(' left ',' @@ ').replace(' right ',' left ').replace(' @@ ',' right ')
PART=np.array([P1.index(sw(p)) if sw(p) in P1 else i for i,p in enumerate(P1)])
def model(kind='ZRu'):
    Z=np.load(os.path.expandvars('$BODY_REPO/hires/v2/')+'emb3_b53.npz')[kind][IDX]; return Z-Z.mean(0)
