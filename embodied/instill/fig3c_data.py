"""Per-touch data for fig3c. Usage: python fig3c_data.py named|nameless SEED
Loads the v3.1 Qwen3-8B checkpoint and writes v3/fig3c_{arm}_s{seed}.npz:
 static touches (2 arms x 23 places x NPOS postures): layer-18 state, verbal position answer;
 posture set (fixed skin spots x many postures): verbal left-right answer vs true world x;
 babble self/other set: verbal answer to 'did you cause this touch yourself'."""
import sys, os, time, numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sim3 as sim, v3

arm_kind, seed = sys.argv[1], int(sys.argv[2]); named = arm_kind == "named"
MODEL = "Qwen/Qwen3-8B"; tag = f"Qwen3-8B_{arm_kind}_s{seed}"
CK = f"$BODY_RUNS{'named' if named else 'namel'}{seed}/ckpt_{tag}.pt"
B = v3.Body(MODEL, named, torch.bfloat16)
ck = torch.load(CK, map_location=v3.DEV, weights_only=False)
B.ad.load_state_dict(ck["ad"]); B.fwd.load_state_dict(ck["fwd"]); B.m.load_state_dict(ck["lora"], strict=False); B.m.eval()
T = v3.T; LAY = B.L_FWD
qs = sim.questions(dict(arm=0, seg=0, arc=.5, x=0, y=0, p=.5, src="other", lh_y=0, rh_y=0), named)
QPOS = [q for k, q, a, c in qs if k == "pos"][0]; QX = [q for k, q, a, c in qs if k == "x"][0]; QSELF = [q for k, q, a, c in qs if k == "self"][0]
t0 = time.time()


def static_ep(q, arm, arc, seed=0):
    A_ = sim.landmark_point(q, arm, arc); env = sim.Env(q, np.random.default_rng(seed))
    s_ = sim.seg_of(arc); P, Q = sim.seg_pts(q)[3 * arm + s_]
    nrm = np.array([-(Q - P)[1], (Q - P)[0]]); nrm /= np.linalg.norm(nrm)
    env.ext = dict(c=A_ + nrm * .04, v=np.zeros(2), r=.1, p=.8, t_end=T - 1)
    S = [env.sense()]
    for t in range(T - 1): env.step(12); S.append(env.sense())
    return dict(S=np.stack(S), A=[12] * (T - 1), kind="babble", g={}, info=sim.summarize(env))


@torch.no_grad()
def states(eps, q, bs=24):
    H, INJ = [], []
    for k in range(0, len(eps), bs):
        ee = eps[k:k + bs]; items = [B.prompt(e, q, None, False) for e in ee]; ids, lab, kind, att, X = B.collate(items, ee)
        out = B.run(ids, kind, att, X, hidden=True); kp = torch.tensor([it[4][-1] for it in items], device=v3.DEV); r = torch.arange(len(ee), device=v3.DEV)
        H.append(out.hidden_states[LAY][r, kp].float().cpu().numpy())
        v = B.ad(X, kind); INJ.append(torch.cat([v[r, kp - 1], v[r, kp]], -1).float().cpu().numpy())
    return np.concatenate(H), np.concatenate(INJ)


def ev_digits(eps, q):
    LP = B.cand_logp(eps, [q] * len(eps), [sim.DIG] * len(eps))
    return np.array([(lambda p: p / p.sum())(np.exp(lp - lp.max())) @ np.arange(11) for lp in LP])


NPOS = int(os.environ.get("NPOS", 12))
rng = np.random.default_rng(42); POS = [sim.rand_q(rng) for _ in range(NPOS)]
ARCS = np.linspace(.03, sim.ARM_LEN - .03, 23)
eps, meta = [], []
for a in range(2):
    for arc in ARCS:
        for pi, q in enumerate(POS): eps.append(static_ep(q, a, arc)); meta.append((a, arc, pi))
meta = np.array(meta)
H, INJ = states(eps, QPOS); print("states", time.time() - t0, flush=True)
EVp = ev_digits(eps, QPOS); print("pos answers", time.time() - t0, flush=True)
truex_s = np.array([e["info"].get("x", np.nan) for e in eps])
# posture set: 3 skin spots per arm (upper arm, forearm, hand) x NP2 postures
NP2 = int(os.environ.get("NP2", 30)); rng = np.random.default_rng(5); P2 = [sim.rand_q(rng) for _ in range(NP2)]
SPOTS = [.25, .72, 1.05]; pe, pm = [], []
for a in range(2):
    for sp in SPOTS:
        for pi, q in enumerate(P2): pe.append(static_ep(q, a, sp)); pm.append((a, sp, pi))
pm = np.array(pm); EVx = ev_digits(pe, QX); truex = np.array([e["info"].get("x", np.nan) for e in pe])
skin_p = np.stack([e["S"][-1][22:] for e in pe])
print("x answers", time.time() - t0, flush=True)
# self vs other on held-out babble episodes
rng = np.random.default_rng(321); be = []
while len(be) < int(os.environ.get("NSELF", 300)):
    e = sim.make_episode(rng, T, kind="babble")
    if e["info"]["arm"] is not None and e["info"]["src"] in ("self", "other"): be.append(e)
LP = B.cand_logp(be, [QSELF] * len(be), [["yes", "no"]] * len(be))
said_yes = np.array([lp[0] > lp[1] for lp in LP]); is_self = np.array([e["info"]["src"] == "self" for e in be])
crossed = np.array([e["info"]["crossed"] for e in be])
print("self", time.time() - t0, (said_yes == is_self).mean(), flush=True)
np.savez(f"v3/fig3c_{arm_kind}_s{seed}.npz", H=H, INJ=INJ, meta=meta, EVp=EVp, truex_s=truex_s, pm=pm, EVx=EVx, truex=truex,
         said_yes=said_yes, is_self=is_self, crossed=crossed, layer=LAY)
print("done", time.time() - t0)
