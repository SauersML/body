import numpy as np, pickle, env as E, episodes as P
orig = E.Sim.__init__; last = {}
def init(s, *a, **k): orig(s, *a, **k); s.rec = []; last["s"] = s
E.Sim.__init__ = init; V = {}; rng = np.random.default_rng(3)
for kind in P.KINDS:
    tr, inf, ok = [], [], []
    for j in range(3):
        e = P.make_episode(rng, kind); tr.append(last["s"].rec); inf.append(e["info"]); ok.append(bool(e["info"]["success"]))
    V[kind] = dict(traj=tr, info=inf, ok=ok)
pickle.dump(V, open("vids_expert.pkl", "wb"))
