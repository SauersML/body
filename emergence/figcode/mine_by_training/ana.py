# "Whose body is touched" score, 95% CI by two-way bootstrap (54 body places x 4 crossed wordings / 8 touch wordings), mean over layers at 50-80% depth.
# Per-sentence projections on the model's own your-minus-woman's direction (own_ci.py, pods hypha-prod-minefig10/11). Point value = emerge.py's own_locus.
import numpy as np, json, glob, os
R = {}
fs = sorted(glob.glob('$BODY_RUNS[013]/*.npz'))
rng = np.random.default_rng(0); B = 4000
for f in fs:
    tag = os.path.basename(f)[:-4]; d = np.load(f)
    for fmt in sorted({k.split('|')[0] for k in d.files if '|proj|' in k}):
        P = {c: d[f'{fmt}|proj|{c}'] for c in ['touch', 'own_her', 'cross_you', 'cross_her']}  # (54, nt, layers)
        sc = lambda pi, t8, t4: float(np.mean((P['cross_you'][pi][:, t4].mean((0, 1)) - P['cross_her'][pi][:, t4].mean((0, 1)))
                                          / (P['touch'][pi][:, t8].mean((0, 1)) - P['own_her'][pi][:, t8].mean((0, 1)))))
        pt = sc(np.arange(54), np.arange(8), np.arange(4))
        bs = [sc(rng.integers(0, 54, 54), rng.integers(0, 8, 8), rng.integers(0, 4, 4)) for _ in range(B)]
        R[f'{tag}|{fmt}'] = dict(m=pt, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        print(f'{tag:16s} {fmt:11s} {pt:6.2f} [{R[f"{tag}|{fmt}"]["lo"]:.2f}, {R[f"{tag}|{fmt}"]["hi"]:.2f}]')
json.dump(R, open(os.path.dirname(os.path.abspath(__file__)) + '/scores.json', 'w'), indent=1)
