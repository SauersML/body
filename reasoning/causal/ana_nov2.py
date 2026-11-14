"""All conditions of out32n merged across the two scenario folds (each condition was fit on one half and tested on the other, so the merged
vector covers every item once, always held out). Cluster bootstrap over scenarios. -> out32n/stats.json"""
import json, os, numpy as np, collections, re
D = "out32n"; I = json.load(open("items_nov.json")); N = len(I)
task = np.array([i["task"] for i in I]); scen = np.array([i["scen"] + (0 if i["fam"] == "body" else 100) for i in I]); twin = np.array([i["twin"] for i in I])
rng = np.random.default_rng(0); GS = ["body_swap", "obj_swap", "body_norm", "obj_norm"]
def arr(x): return np.array([np.nan if v is None else v for v in x], float)
R = {k: arr(v) for k, v in json.load(open(f"{D}/margins.json")).items()}
P = json.load(open(f"{D}/patch.json")) if os.path.exists(f"{D}/patch.json") else {}
DAS = json.load(open(f"{D}/das.json")) if os.path.exists(f"{D}/das.json") else {}
base = R["base"]
def merged(src, pat):  # pat contains {f}
    a, b = src.get(pat.format(f=0)), src.get(pat.format(f=1))
    if a is None or b is None: return None
    a, b = arr(a) if not isinstance(a, np.ndarray) else a, arr(b) if not isinstance(b, np.ndarray) else b
    return np.where(np.isnan(a), b, a)
B = 1000; BW = []
for _ in range(B):
    w = collections.Counter()
    for side in ["body", "obj"]: s = np.unique(scen[np.char.startswith(task, side)]); w.update(rng.choice(s, len(s)))
    BW.append(np.array([w.get(s, 0) for s in scen], float))
BW = np.array(BW)  # [B, N] item weights
def wm(x, mask, W):
    ok = mask & ~np.isnan(x); return (W[..., ok] * x[ok]).sum(-1) / W[..., ok].sum(-1)
ONE = np.ones(N)
def eff(m, kind="acc"):
    """per group: delta vs base (acc or margin), CI; plus body-minus-object swap selectivity"""
    f = (lambda x: (x > 0).astype(float)) if kind == "acc" else (lambda x: x)
    d = f(m) - f(base); d[np.isnan(m)] = np.nan; out = {}
    for g in GS:
        bs = wm(d, task == g, BW); out[g] = [round(float(wm(d, task == g, ONE)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    sel = wm(d, task == "body_swap", BW) - wm(d, task == "obj_swap", BW); s0 = wm(d, task == "body_swap", ONE) - wm(d, task == "obj_swap", ONE)
    out["sel"] = [round(float(s0), 3), round(float(np.percentile(sel, 2.5)), 3), round(float(np.percentile(sel, 97.5)), 3), round(float(min(1, 2 * min((sel <= 0).mean(), (sel >= 0).mean()))), 4)]
    return out
S = {}
# baseline
S["base_acc"] = {g: round(float((base[task == g] > 0).mean()), 3) for g in GS}
bacc = (base > 0).astype(float); g = wm(bacc, task == "body_swap", BW) - wm(bacc, task == "obj_swap", BW)
S["base_gap"] = [round(float(wm(bacc, task == "body_swap", ONE) - wm(bacc, task == "obj_swap", ONE)), 3), round(float(np.percentile(g, 2.5)), 3), round(float(np.percentile(g, 97.5)), 3), float(2 * min((g <= 0).mean(), 1))]
# swap shift: how far the swap moves the answer toward the displaced part (swap margin + twin's margin), and the habit prior (twin's margin)
shift = base + base[twin]; S["shift"] = {}
for side in ["body", "obj"]:
    msk = task == f"{side}_swap"; S["shift"][side] = dict(shift=round(float(shift[msk].mean()), 2), prior=round(float(base[twin][msk].mean()), 2), swap_margin=round(float(base[msk].mean()), 2))
gs = wm(shift, task == "body_swap", BW) - wm(shift, task == "obj_swap", BW); S["shift"]["body_minus_obj_CI"] = [round(float(np.percentile(gs, q)), 2) for q in (2.5, 97.5)]
# does side still matter once the habit prior is matched? scenario-level regression swap_margin ~ prior + side
X, y = [], []
for s in np.unique(scen[np.char.endswith(task, "swap")]):
    msk = (scen == s) & np.char.endswith(task, "swap"); X.append([1, base[twin][msk].mean(), float(s < 100)]); y.append(base[msk].mean())
X, y = np.array(X), np.array(y); coef = np.linalg.lstsq(X, y, rcond=None)[0]
bc = []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y)); bc.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
bc = np.array(bc); S["regress_swapmargin"] = dict(prior=[round(float(coef[1]), 3)] + [round(float(np.percentile(bc[:, 1], q)), 3) for q in (2.5, 97.5)],
                                                  body=[round(float(coef[2]), 2)] + [round(float(np.percentile(bc[:, 2], q)), 2) for q in (2.5, 97.5)])
# directions: ablations and additions
S["dirs"] = {}
names = sorted({re.sub(r"^f[01]_", "", k) for k in R if k.startswith("f0_")})
for nm in names:
    m = merged(R, "f{f}_" + nm)
    if m is None: continue
    S["dirs"][nm] = dict(acc=eff(m, "acc"), margin=eff(m, "margin"))
# patching: components ranked on the other fold; 'twin' = put the normal twin's activations into the swap item; 'mean' = mean-ablate
S["patch"] = {}
for nm in sorted({re.sub(r"^f[01]_", "", k) for k in P if k.startswith("f0_")}):
    m = merged(P, "f{f}_" + nm)
    if m is None: continue
    e = dict(acc=eff(m, "acc"), margin=eff(m, "margin"))
    if nm.startswith("twin"):  # fraction of the swap shift removed, per side
        frac = (base - m) / np.where(shift > 0.5, shift, np.nan); e["frac_removed"] = {}
        for side in ["body", "obj"]:
            bs = wm(frac, task == f"{side}_swap", BW); e["frac_removed"][side] = [round(float(wm(frac, task == f"{side}_swap", ONE)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        d = wm(frac, task == "body_swap", BW) - wm(frac, task == "obj_swap", BW); e["frac_removed"]["p_body_vs_obj"] = round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)
    S["patch"][nm] = e
if "rank" in P:  # overlap of top components between body and object attributions
    for f in ["f0", "f1"]:
        bA, oA = np.array(P["rank"][f]["bodyA"]), np.array(P["rank"][f]["objA"])
        tb, to = set(np.argsort(bA)[:20]), set(np.argsort(oA)[:20]); S.setdefault("attr_overlap_top20", {})[f] = len(tb & to)
        S.setdefault("attr_corr", {})[f] = round(float(np.corrcoef(bA, oA)[0, 1]), 3)
        S.setdefault("attr_top", {})[f] = [(int(c // 65 + 1), "MLP" if c % 65 == 64 else int(c % 65), round(float(bA[c]), 2), round(float(oA[c]), 2)) for c in np.argsort(bA)[:10]]
# DAS
S["das"] = {}
for key in sorted({re.sub(r"^f[01]_", "", k) for k in DAS}):
    if f"f0_{key}" not in DAS or f"f1_{key}" not in DAS: continue
    e = {}
    for nm in ["learned", "rand0", "rand1", "rand2"]:
        m = merged({k: v[nm] for k, v in DAS.items() if nm in v}, "f{f}_" + key); e[nm] = dict(acc=eff(m, "acc"), margin=eff(m, "margin"))
    e["cos"] = {kk: round((DAS[f"f0_{key}"]["cos"][kk] + DAS[f"f1_{key}"]["cos"][kk]) / 2, 2) for kk in DAS[f"f0_{key}"]["cos"]}
    S["das"][key] = e
json.dump(S, open(f"{D}/stats.json", "w"), indent=1)
print(json.dumps({k: S[k] for k in ["base_acc", "base_gap", "shift", "regress_swapmargin"]}))
