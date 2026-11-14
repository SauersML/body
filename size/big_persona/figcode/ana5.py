"""Rows for big_persona_bigger: top 12 traits by |bigger85 - unsteered| in % answered trait-matching way; 95% CI paired bootstrap within halves;
q = BH-FDR of balanced logit test vs unsteered (ana.py). rand = same change for the 3 random directions at norm 85."""
import json, os, numpy as np
os.chdir('$BODY_WORK/fig/bigp2')
exec(open('ana.py').read().split('rng =')[0].split('out = {}')[0])
e0, p0 = bal('big85', 'base'); q0 = bh(p0)
rng = np.random.default_rng(0)
mb = match('base'); ms = match('big85'); mr = [match(f'rand{j}_85') for j in range(3)]
D = np.array([100 * (ms - mb)[tr == t].mean() for t in T])
DR = np.array([[100 * (m - mb)[tr == t].mean() for t in T] for m in mr])
print('bigger |d| top12 min', np.sort(abs(D))[-12], 'max', abs(D).max())
for j in range(3): print('rand', j, 'top12 min', np.sort(abs(DR[j]))[-12], 'max', abs(DR[j]).max(), 'q<.05', int((bh(bal(f"rand{j}_85",'base')[1])<.05).sum()))
print('bigger q<.05', int((q0 < .05).sum()))
rows = []
for i in np.where(abs(D) >= 10)[0]:  # top 12 by |d| ties at 10, so all 17 with |d| >= 10
    t = T[i]; k = np.where(tr == t)[0]; ky = k[my[k]]; kn = k[~my[k]]; d = 100 * (ms - mb)
    bs = [np.r_[d[rng.choice(ky, len(ky))], d[rng.choice(kn, len(kn))]].mean() for _ in range(10000)]
    rows.append(dict(trait=t, d=float(D[i]), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                     rand=[float(x) for x in DR[:, i]], q=float(q0[i]), eff=float(e0[i])))
rows.sort(key=lambda r: (r['d'], r['eff']))
for r in rows: print(f"{r['trait'][:60]:60s} {r['d']:+6.1f} [{r['lo']:+.1f},{r['hi']:+.1f}] rand {[round(x) for x in r['rand']]} q {r['q']:.2f}")
json.dump(rows, open('$BODY_WORK/fig/bigp5/rows.json', 'w'), indent=1)
