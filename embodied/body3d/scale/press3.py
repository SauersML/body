"""Felt-pressure steering on the 3D body (reach task). python press3.py MODEL ARM ; env CKDIR (ckpt_{tag}.pt) NP=200 NS=48
Direction: at L_FWD, skin tokens of expert reach episodes (closed-loop expert, diverse starts): mean state over the steps with the
highest summed skin input minus the steps with none (contact vs no contact), unit norm, scaled by the s.d. of the projection.
Test: closed-loop reach on NS diverse scenes, adding k*sd (k in -6,-3,0,3,6, plus a random direction at 6) at every skin token.
Outcomes: success (touches the named part in the last 6 steps), final fingertip distance to the part, summed hand-part contact force."""
import sys, os
os.environ.setdefault("DO", "")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze3d.py")).read()
src = src.split("# ---------------- representations from poke episodes")[0]
exec(compile(src, "analyze3d_head", "exec"))
import pickle, episodes2 as Q
NP_ = int(os.environ.get("NP", 200)); NS = int(os.environ.get("NS", 48)); Ls = B.L_FWD
for L_ in [Ls]:
    if L_ not in LAY: B.layers[L_ - 1].register_forward_hook(steer_hook(L_))
rng = np.random.default_rng(31); eps = [Q.run_expert("reach", *Q.make_scene("reach", rng)) for _ in range(NP_)]
items = [B.tk.prompt(e, []) for e in eps]; Z, Y = [], []
with torch.no_grad():
    for k in range(0, len(items), 24):
        it, ee = items[k:k + 24], eps[k:k + 24]; ids, lab, kind, att, X = TR.collate(it, ee, B.tok.pad_token_id)
        o = B.run(ids, kind, att, X, hidden=True); h = o.hidden_states[Ls]
        for b, (e, x) in enumerate(zip(ee, it)):
            sk = e["S"][:, TR.SL[1]].sum(1)
            for t, p in enumerate(x[4]): Z.append(h[b, p + 1].float().cpu().numpy()); Y.append(sk[t])
Z, Y = np.stack(Z), np.array(Y); hi = Y > np.quantile(Y[Y > 0], 0.5) if (Y > 0).sum() > 20 else Y > 0; lo = Y == 0
v = Z[hi].mean(0) - Z[lo].mean(0); vn = v / np.linalg.norm(v); sd = float((Z @ vn).std())
r_fit = float(np.corrcoef(Z @ vn, Y)[0, 1]); print("pressure dir: n hi/lo", hi.sum(), lo.sum(), "sd", round(sd, 2), "r(proj, skin sum)", round(r_fit, 3), flush=True)
rr = np.random.default_rng(5).standard_normal(vn.shape); rr /= np.linalg.norm(rr)


def run(vec):
    scenes = [Q.make_scene("reach", np.random.default_rng([8800, i]), diverse=True) for i in range(NS)]
    sims = [s[0] for s in scenes]; infos = [s[1] for s in scenes]; S = [[s.sense()] for s in sims]; tc = [[] for _ in sims]; force = np.zeros(NS)
    for t in range(T):
        its = [B.tk.prompt(dict(kind="reach", info=infos[b], S=np.stack(S[b])), [], upto_t=t) for b in range(NS)]
        ids, lab, kind_, att, X = TR.collate(its, [dict(S=np.stack(x)) for x in S], B.tok.pad_token_id)
        if vec is not None:
            mask = torch.zeros(ids.shape, device=DEV)
            for b, it in enumerate(its):
                for p in it[4]: mask[b, p + 1] = 1
            STEER[Ls] = (torch.tensor(vec, dtype=torch.float32).to(DEV), mask)
        with torch.no_grad():
            o = B.run(ids, kind_, att, X); r = torch.arange(NS); a = torch.tensor([it[3][-1] for it in its])
            act = decode(B.act(o.last_hidden_state[r, a.to(DEV)].float()).cpu().numpy(), [x[-1] for x in S])
        STEER.clear()
        for b, sim in enumerate(sims):
            S[b].append(sim.step(np.clip(act[b], H.JR[:, 0] - 0.3, H.JR[:, 1] + 0.3), t)); tc[b].append(E.touch_info(sim))
            if t >= T - 6:
                hg = set(E.HAND_GEOMS["left"] + E.HAND_GEOMS["right"])
                force[b] += sum(f for g1, g2, f in sim.last_pairs if (g1 in hg and H.GPART.get(g2) == infos[b]["part"]) or (g2 in hg and H.GPART.get(g1) == infos[b]["part"]))
    ok = [Q.success("reach", infos[b], sims[b], tc[b]) for b in range(NS)]; sp = [Q.spot_cm("reach", infos[b], sims[b]) for b in range(NS)]
    return dict(ok=[bool(x) for x in ok], spot_cm=sp, force=(force / 6).tolist())


RES = dict(sd=sd, r_fit=r_fit, layer=Ls)
for k in (0, 3, 6, -3, -6):
    RES[f"k{k}"] = run(None if k == 0 else vn * sd * k)
    x = RES[f"k{k}"]; print(f"k={k}: success {np.mean(x['ok']):.3f} spot {np.mean(x['spot_cm']):.1f}cm force {np.mean(x['force']):.2f}", flush=True)
RES["rand6"] = run(rr * sd * 6); x = RES["rand6"]; print(f"rand6: success {np.mean(x['ok']):.3f} spot {np.mean(x['spot_cm']):.1f}cm force {np.mean(x['force']):.2f}", flush=True)
json.dump(RES, open(f"{OUT}/press3_{tag}.json", "w")); print("PRESS3 DONE", flush=True)
