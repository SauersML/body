import sys, json, numpy as np
sys.path.insert(0, '$BODY_WORK/fig/selfother2')
from anchors import P, A
import bodyrender as br
W, Hh, yc, hh = 1680, 2800, 0.15, 8.75
cols = np.tile(np.array([0.80, 0.79, 0.77]), (len(P), 1))
br.render(cols, 'body_hi.png', W=W, Hh=Hh, yc=yc, hh=hh)
hw = hh * W / Hh
proj = {k: [float((v[0] + hw) / (2 * hw) * W), float((yc + hh - v[1]) / (2 * hh) * Hh)] for k, v in A.items()}
json.dump(dict(W=W, H=Hh, a=proj), open('anchors_hi.json', 'w'))
print(proj['lips'], proj['palmL'], proj['toesR'])
