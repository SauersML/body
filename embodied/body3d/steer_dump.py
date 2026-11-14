"""Steering at L_FWD with per-episode records (fingertip paths, qpos, final distance) + random-direction control.
python steer_dump.py MODEL ARM ; env DO=rep NP=400 CKDIR=..."""
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze3d.py")).read()
src = src.split("# ---------------- causal steering")[0]
exec(compile(src, "analyze3d_head", "exec"))
import pickle
targets = ["left thigh", "right thigh", "belly", "left forearm", "right shin", "head", "chest", "left shin"]
Ls = B.L_FWD; Zs = HL[Ls]; mu = Zs.mean(0); scale = np.median(np.linalg.norm(Zs - mu, axis=1))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0.25,0.5,1,1.5,2").split(",")]
rr = np.random.default_rng(99)
def run(sv, p):
    sims, eps, tc, traj = closed_loop("poke_reach", 16, 11, steer=sv, probe=False, record=True)
    recs = []
    for s, e, t, tr in zip(sims, eps, tc, traj):
        Pw, _ = H.taxel_world(s.d)
        dh = {h: float(np.linalg.norm(Pw[H.TPART == p] - s.d.site(f"tip_{h}").xpos, axis=1).min()) for h in ("left", "right")}
        recs.append(dict(dist=min(dh.values()), dist_hand=dh, touch=any(p in x[0] for x in t[-8:]),
                         qpos=np.stack([x["qpos"] for x in tr]).astype(np.float32), info=e["info"]))
    return recs
OUTD = {}
for nmx in targets:
    p = H.PARTS.index(nmx); v = Zs[part == p].mean(0) - mu; vn = v / np.linalg.norm(v)
    r = rr.standard_normal(v.shape); r = r / np.linalg.norm(r)
    OUTD[nmx] = {"none": run(None, p)}
    for a in ALPHAS:
        OUTD[nmx][f"a{a}"] = run((Ls, torch.tensor(vn * scale * a, dtype=torch.float32), [1, 2, 3, 4]), p)
    OUTD[nmx]["rand1.5"] = run((Ls, torch.tensor(r * scale * 1.5, dtype=torch.float32), [1, 2, 3, 4]), p)
    print("steer", nmx, {k: round(100 * float(np.mean([x["dist"] for x in v_])), 1) for k, v_ in OUTD[nmx].items()}, round(time.time() - t00), flush=True)
pickle.dump(dict(steer=OUTD, layer=Ls, scale=float(scale), alphas=ALPHAS, mean_vec_norm=float(np.linalg.norm(mu))), open(f"{OUT}/steer_dump_{tag}.pkl", "wb"))
print("STEER DUMP DONE", flush=True)
