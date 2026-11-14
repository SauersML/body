"""Empirical canonical targets: per part, try candidate taxels (hand by rule) from 8 varied start poses; keep each taxel's success rate."""
import sys,os; sys.path.insert(0,'..'); os.environ['X2_CANON']='0'
import numpy as np, json, env as E, hum as H, episodes2 as Q, expert2 as X
from multiprocessing import Pool
def score(args):
    part, ti = args; h = X.rule_hand(part); ok = 0
    for k in range(8):
        sim = Q.start_sim(np.random.default_rng([77, k]), True); pol = X.Reach(h, ti, part); tc = []
        for t in range(20): sim.step(sim.grav_ctrl(pol.act(sim, t)), t); tc.append(E.touch_info(sim))
        ok += any(part in x[0] for x in tc[-6:])
    return part, ti, ok / 8
if __name__ == "__main__":
    jobs = []
    for part in range(len(H.PARTS)):
        sim = E.Sim("mount", None); P, N = H.taxel_world(sim.d); ix = np.where(H.TPART == part)[0]; h = X.rule_hand(part)
        tip = sim.d.site(f"tip_{h}").xpos; v = tip - P[ix]; dist = np.linalg.norm(v, axis=1); face = (N[ix] * v).sum(1) / dist
        sc = dist - 0.15 * face - 3.0 * X.EDGE[ix]; cand = ix[np.argsort(sc)[:min(24, len(ix))]]
        jobs += [(part, int(t)) for t in cand]
    with Pool(14) as p: res = p.map(score, jobs)
    out = {}
    for part, ti, s in res: out.setdefault(str(part), {})[str(ti)] = s
    json.dump(out, open("canon_scores.json", "w"))
    for part in range(len(H.PARTS)):
        d = out[str(part)]; b = max(d, key=d.get); print(H.PARTS[part], "best", b, d[b], "n>=.75:", sum(v >= .75 for v in d.values()))
