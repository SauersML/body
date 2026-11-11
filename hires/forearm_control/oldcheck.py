import os
import json, numpy as np, fana as A, fstim as S
r = json.load(open(os.path.expandvars('$BODY_RUNS/body32c/raw_short/rows.json')))['rows']
X = np.load(os.path.expandvars('$BODY_RUNS/body32c/raw_short/reply.npy'), mmap_mode='r')
want = {p: si for c, si, v, p in S.phrases() if c == "orig"}
idx = collections = {}
from collections import defaultdict
rowsby = defaultdict(list)
for i, x in enumerate(r):
    if x['kind'] == 'loc' and x['phrase'] in want: rowsby[want[x['phrase']]].append(i)
assert len(rowsby) == 80
for li, L in enumerate([50, 53, 57]):
    V = np.stack([X[li, rowsby[s]].astype(np.float32).mean(0) for s in range(80)])
    for arm in ["left"]:
        idx = np.where(A.ARM == arm)[0]; D = A.dmat(V[idx])
        print(L, arm, {k: round(v, 3) for k, v in A.metrics(D, idx).items()})
print("chance", A.chance(idx))
