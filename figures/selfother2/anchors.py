import os
# 54 body places (somato stimuli keys) -> anchor points on neutral.obj; per-vertex soft blend of place values
import sys, numpy as np
sys.path.insert(0, os.path.expandvars('$BODY_REPO/figures/zoom3'))
from load import load, joints
H = os.path.expandvars('$BODY_REPO/mesh/')
V, G = load(H + 'base.obj'); J = joints(V, G)
Vn, Gn = load(H + 'neutral.obj'); V = V.copy(); V[:len(Vn)] = Vn
faces = Gn['body'] if 'body' in Gn else G['body']
used = np.unique(np.concatenate([np.array(f) for f in faces])); rm = -np.ones(len(V), int); rm[used] = np.arange(len(used)); P = V[used]
TRI = []
for f in faces:
    f = [rm[i] for i in f]; TRI += [[f[0], f[1], f[2]], [f[0], f[2], f[3]]] if len(f) == 4 else [f]
TRI = np.array(TRI)
# vertex normals
vn = np.zeros_like(P); fn = np.cross(P[TRI[:, 1]] - P[TRI[:, 0]], P[TRI[:, 2]] - P[TRI[:, 0]])
for k in range(3): np.add.at(vn, TRI[:, k], fn)
NRM = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)
def near(pt, dirn=None, rad=0.6):
    """surface point near pt; if dirn, the vertex within rad whose normal best matches dirn"""
    d = np.linalg.norm(P - pt, axis=1)
    if dirn is None: return P[np.argmin(d)]
    c = np.argsort(d)[:int(400 * rad)]; sc = NRM[c] @ (np.array(dirn) / np.linalg.norm(dirn)) - 0.6 * d[c] / d[c].max()
    return P[c[np.argmax(sc)]]
A = {}; GRP = {}
head_c = (J['head'] + J['head-2']) / 2 if 'head-2' in J else J['head']
for s, sg in (('l', 'L'), ('r', 'R')):
    j = lambda n: J[f'{s}-{n}']; x = 1 if s == 'l' else -1
    for k, nm in enumerate(['thumb', 'index', 'middle', 'ring', 'little']):
        A[nm + sg] = near((j(f'finger-{k+1}-3') + j(f'finger-{k+1}-4')) / 2); GRP[nm + sg] = 'hand' + sg
    hc = np.mean([j(f'finger-{k}-1') for k in range(2, 6)], 0) * 0.6 + j('hand') * 0.4
    HV = P[np.linalg.norm(P - hc, axis=1) < 0.45]; u, sv, vt = np.linalg.svd(HV - HV.mean(0)); nrm = vt[2] * (1 if vt[2][1] < 0 else -1)  # palm faces down
    A['palm' + sg] = near(hc, nrm); A['backhand' + sg] = near(hc, -nrm); GRP['palm' + sg] = GRP['backhand' + sg] = 'hand' + sg
    A['wrist' + sg] = near(j('hand')); A['forearm' + sg] = near((j('elbow') + j('hand')) / 2, [0, 0, 1])
    A['elbow' + sg] = near(j('elbow'), [0, 0, -1]); A['upperarm' + sg] = near((j('shoulder') + j('elbow')) / 2, [0, 0, 1])
    A['shoulder' + sg] = near(j('shoulder') + [x * 0.2, 0.35, 0], [x * 0.3, 1, 0])
    for k in ['wrist', 'forearm', 'elbow', 'upperarm', 'shoulder']: GRP[k + sg] = 'arm' + sg
    toes = np.mean([j(f'toe-{k}-2') for k in range(1, 6)], 0)
    A['toes' + sg] = near(toes, [0, 0.3, 1]); A['sole' + sg] = near((toes + j('ankle')) / 2 + [0, -0.4, 0], [0, -1, 0], 0.9)
    A['heel' + sg] = near(j('ankle') + [0, -0.45, -0.3], [0, -0.3, -1], 0.9); A['ankle' + sg] = near(j('ankle'), [x, 0, 0])
    for k in ['toes', 'sole', 'heel', 'ankle']: GRP[k + sg] = 'foot' + sg
    A['shin' + sg] = near((j('knee') + j('ankle')) / 2, [0, 0, 1]); A['knee' + sg] = near(j('knee'), [0, 0, 1])
    A['thigh' + sg] = near((j('upper-leg') + j('knee')) / 2, [0, 0, 1]); A['hip' + sg] = near(j('upper-leg') + [x * 0.5, 0.3, 0], [x, 0, 0])
    for k in ['shin', 'knee', 'thigh', 'hip']: GRP[k + sg] = 'leg' + sg
    A['cheek' + sg] = near(head_c + [x * 0.45, -0.25, 0.5], [x * 0.5, 0, 1]); A['ear' + sg] = near(head_c + [x * 0.8, 0, 0], [x, 0, 0])
    GRP['cheek' + sg] = GRP['ear' + sg] = 'head'
A['forehead'] = near(head_c + [0, 0.45, 0.7], [0, 0.3, 1]); A['crown'] = near(head_c + [0, 1.2, 0], [0, 1, 0])
A['lips'] = near(head_c + [0, -0.55, 0.8], [0, 0, 1]); A['tongue'] = A['lips']
A['neckF'] = near(J['neck'] + [0, -0.1, 0.4], [0, 0, 1]); A['neckB'] = near(J['neck'] + [0, 0.1, -0.4], [0, 0, -1])
for k in ['forehead', 'crown', 'lips', 'tongue', 'neckF', 'neckB']: GRP[k] = 'head'
sp = [J[f'spine-{k}'] for k in range(1, 5)]
A['chest'] = near(J['spine-1'] + [0, -0.2, 1], [0, 0, 1]); A['belly'] = near(J['spine-3'] + [0, 0, 1], [0, 0, 1])
A['upback'] = near(J['spine-1'] + [0, -0.2, -1], [0, 0, -1]); A['lowback'] = near(J['spine-3'] + [0, 0, -1], [0, 0, -1])
for k in ['chest', 'belly', 'upback', 'lowback']: GRP[k] = 'torso'
KEYS = list(A); AP = np.array([A[k] for k in KEYS])
# vertex group: nearest anchor's group, but hands/feet decided by distance to their anchors
dist = np.linalg.norm(P[:, None] - AP[None], axis=2)
SIG = {'hand': 0.25, 'foot': 0.35, 'head': 0.45, 'arm': 0.7, 'leg': 0.9, 'torso': 1.2}
sig = np.array([SIG[GRP[k].rstrip('LR')] for k in KEYS])
W = np.exp(-(dist / sig[None]) ** 2)
def colorfield(vals):
    """vals: dict key->value. returns per-vertex value (soft blend of nearest places)"""
    v = np.array([vals[k] for k in KEYS]); w = W + 1e-12
    return (w @ v) / w.sum(1)
if __name__ == '__main__':
    print(len(KEYS)); print({k: A[k].round(2).tolist() for k in ['toesL', 'palmL', 'backhandL', 'heelL', 'crown', 'chest']})
