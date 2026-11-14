"""First interp on the trained body: somatotopy (RSA), touch-in-world integration probe, steering along the arm."""
import json, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
import body, train
from train import T, DEV

torch.manual_seed(0)
B = train.Body(); ck = torch.load("ckpt.pt")
B.ad.load_state_dict(ck["ad"]); B.fwd.load_state_dict(ck["fwd"])
B.m.load_state_dict(ck["lora"], strict=False); B.m.eval()
B_rand = train.Adapter().to(DEV)  # untrained adapter control
LAYERS = [6, 10, 14, 20, 27]
res = {}


@torch.no_grad()
def states(S_list, q="Where are you being touched?"):
    """Hidden state at the last body token for each layer + the injected vector."""
    ids, lab, phm, att, Sx = B.build(S_list, [q] * len(S_list))
    out = B.run(ids, phm, att, Sx, hidden=True)
    return {l: out.hidden_states[l][:, T].float().cpu().numpy() for l in LAYERS}, B.ad(Sx)[:, T].cpu().numpy()


# 1. Somatotopy: one-hot touch on each of 24 sensors (no geometry in the input), averaged over 20 random poses
rng = np.random.default_rng(1); poses = rng.uniform(-1.2, 1.2, (20, 6))
G = body.skin_geodesic(); iu = np.triu_indices(body.N_SKIN, 1)
reps = {l: np.zeros((body.N_SKIN, 1024)) for l in LAYERS + ["inject", "rand_adapter"]}
for q in poses:
    S = []
    for i in range(body.N_SKIN):
        tch = np.zeros(body.N_SKIN); tch[i] = 1.0
        S.append(np.tile(body.sensors(q, tch), (T, 1)))
    H, inj = states(S)
    with torch.no_grad():
        rv = B_rand(torch.tensor(np.stack(S), device=DEV))[:, -1].cpu().numpy()
    for l in LAYERS: reps[l] += H[l] / len(poses)
    reps["inject"] += inj / len(poses); reps["rand_adapter"] += rv / len(poses)
def cosd(X):
    X = X - X.mean(0); X = X / np.linalg.norm(X, axis=1, keepdims=True); return 1 - X @ X.T
rsa = {str(k): float(spearmanr(cosd(v)[iu], G[iu])[0]) for k, v in reps.items()}
res["somatotopy_spearman_vs_skin_geodesic"] = rsa; print("RSA", rsa)

# 2. Integration: is touch location coded in WORLD coordinates (needs pose + skin site)? Linear probes, 5-fold CV R^2.
rng = np.random.default_rng(2); S, Y = [], []
while len(S) < 600:
    s, info = body.episode(rng, T, p_touch=1.0)
    if info["touch_world"] is not None: S.append(s); Y.append(info["touch_world"])
S, Y = np.stack(S), np.stack(Y); Hs = {l: [] for l in LAYERS}; Inj = []
for k in range(0, len(S), 50):
    H, inj = states(list(S[k:k + 50]))
    for l in LAYERS: Hs[l].append(H[l])
    Inj.append(inj)
r2 = lambda X: float(np.mean(cross_val_score(RidgeCV(alphas=np.logspace(-2, 4, 13)), X, Y[:, 1], cv=5, scoring="r2")))
integ = {"raw_sensors": r2(S[:, -1]), "inject": r2(np.concatenate(Inj))}
integ.update({f"L{l}": r2(np.concatenate(Hs[l])) for l in LAYERS})
# control: skin site alone (which sensor, one-hot) without pose
site = np.zeros((len(S), body.N_SKIN)); site[np.arange(len(S)), S[:, -1, 12:].argmax(1)] = 1
integ["skin_site_only"] = r2(site)
res["touch_world_y_R2"] = integ; print("integration R2", integ)

# 3. Steering: interpolate the injected last-step vector between 'touch on left forearm sensor' and 'touch on left hand sensor'
#    in a body that is NOT touched; read which part it reports.
rng = np.random.default_rng(3); base = []
while len(base) < 40:
    s, info = body.episode(rng, T, p_touch=0.0)
    if info["part"] == "nowhere": base.append(s)
def mean_inject(sensor_idx):
    v = []
    for s in base:
        s2 = s.copy(); s2[-1, 12:] = 0; s2[-1, 12 + sensor_idx] = 1.0
        v.append(B.ad(torch.tensor(s2[-1], device=DEV)).detach())
    return torch.stack(v).mean(0)
with torch.no_grad():
    v0 = torch.stack([B.ad(torch.tensor(s[-1], device=DEV)) for s in base]).mean(0)
idx_from, idx_to = 0 * 4 + 1, 2 * 4 + 2  # left upper arm sensor -> left hand sensor
vA, vB = mean_inject(idx_from) - v0, mean_inject(idx_to) - v0
alphas = np.linspace(0, 1, 6); steer = []
for a in alphas:
    vec = (1 - a) * vA + a * vB
    def extra_for(n_tokens):
        e = torch.zeros(1, n_tokens, 1024, device=DEV); e[0, T] = vec; return e
    ids, lab, phm, att, Sx = B.build([base[0]] * 7, ["Where are you being touched?"] * 7, body.PARTS + ["nowhere"])
    B.extra = extra_for(ids.shape[1])
    P = train.score(B, base[:20], ["Where are you being touched?"] * 20, [body.PARTS + ["nowhere"]] * 20)
    B.extra = None
    steer.append({p: P.count(p) / len(P) for p in set(P)}); print(a, steer[-1])
res["steer_upperarm_to_hand"] = dict(alphas=alphas.tolist(), reports=steer)
json.dump(res, open("interp.json", "w"), indent=1)

# figure
fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
ax[0].imshow(G, cmap="viridis"); ax[0].set_title("skin geodesic distance\n(sensors: L upper,fore,hand | R ...)")
best = max(LAYERS, key=lambda l: rsa[str(l)])
ax[1].imshow(cosd(reps[best]), cmap="viridis"); ax[1].set_title(f"model distance, layer {best}\nspearman={rsa[str(best)]:.2f}")
ks = ["rand_adapter", "inject"] + LAYERS
ax[2].bar([str(k) for k in ks], [rsa[str(k)] for k in ks]); ax[2].set_title("somatotopy (RSA vs skin geodesic)")
ks2 = ["skin_site_only", "raw_sensors", "inject"] + [f"L{l}" for l in LAYERS]
ax[3].bar(ks2, [integ[k] for k in ks2]); ax[3].set_title("touch height in world: linear R²"); ax[3].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("fig_interp.png", dpi=110)
