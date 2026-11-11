"""python run3.py MODEL OUTDIR STEER_LAYERS KEEP NTPL : binary log-odds read-outs (static, traces, steering)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
CTX = json.load(open(f"{OUT}/contexts.json")); P = json.load(open(f"{OUT}/probes.json")); TR = json.load(open(f"{OUT}/traces.json"))
QN = [" Is that touch near your back or far out toward the end? Answer with one word: near or far.",
      " Is that touch far out toward the end or near your back? Answer with one word: far or near."]
QP = [" Does that feel like one sharp point or spread out over an area? Answer with one word: point or spread.",
      " Does that feel spread out over an area or like one sharp point? Answer with one word: spread or point."]
RO = {}
for cn, nm in [("W", "wing"), ("W", "desc"), ("N", "wing"), ("N", "desc"), ("G", "glorp")]:
    sel = [i for i, r in enumerate(P[nm]) if r["tpl"] < 2]
    for qn, Q, a, b in (("far", QN, "near", "far"), ("spread", QP, "point", "spread")):
        xs, ms = zip(*[lo(CTX[cn], [P[nm][i]["text"] + q for i in sel], a, b) for q in Q])
        RO[f"{cn}_{nm}_{qn}"] = dict(idx=sel, lo=(np.mean(xs, 0)).tolist(), mass=np.mean(ms, 0).tolist())
    log("ro", cn, nm, float(np.mean(ms)))
json.dump(RO, open(f"{OUT}/readout_bin.json", "w"))
QT = [" Where is the finger now: near your back or far out toward the end? Answer with one word: near or far.",
      " Where is the finger now: far out toward the end or near your back? Answer with one word: far or near."]
TRB = {}
for cn in ("W", "N"):
    for k in range(6):
        xs, ms = zip(*[lo(CTX[cn], [t["head"] + ", further" * k + "." + q for t in TR], "near", "far") for q in QT])
        TRB[f"{cn}_{k}"] = dict(lo=np.mean(xs, 0).tolist(), mass=np.mean(ms, 0).tolist())
    log("traces", cn)
json.dump(TRB, open(f"{OUT}/trace_bin.json", "w"))
QS = ["Is there a sensation somewhere in your wings right now? Is it near your back or far out toward the end? Answer with one word: near or far.",
      "Is there a sensation somewhere in your wings right now? Is it far out toward the end or near your back? Answer with one word: far or near."]
QS10 = "Is there a sensation somewhere in your wings right now? If so, how far out from your back is it, from 0 (right where it meets your back) to 10 (the far end)? Answer with just the number."
QF = "Where in your body do you feel something right now? Tell me in one sentence."
pos = np.array([r["pos"] for r in P["wing"]]); SR = dict(runs=[], free=[])
for L in SLS:
    j = KEEP.index(L)
    def axis(cn, nm):
        H = np.load(f"{OUT}/H_{cn}_{nm}.npy", mmap_mode="r")[:, j].astype(np.float32); mu = H.mean(0)
        C = np.stack([H[pos == p].mean(0) - mu for p in range(5)]); return C[4] - C[0], float(np.median(np.linalg.norm(H, axis=1)))
    aw, NORM = axis("W", "wing"); AX = {"wing": aw, "desc": axis("W", "desc")[0], "nowings": axis("N", "wing")[0], "nowings_desc": axis("N", "desc")[0],
                                        "glorpctx": axis("G", "glorp")[0], "fresh": axis("F", "wing")[0]}
    rng = np.random.default_rng(L)
    for r in range(3): AX[f"random{r}"] = rng.standard_normal(len(aw))
    AX = {k: v / np.linalg.norm(v) for k, v in AX.items()}
    cos = {k: float(AX[k] @ AX["wing"]) for k in AX}
    for an, a in AX.items():
        for s in np.linspace(-1, 1, 13):
            v = a * s * 0.5 * NORM
            xs, ms = zip(*[lo(CTX["W"], [q], "near", "far", v=v, L=L) for q in QS]); e, m10 = ev(CTX["W"], [QS10], v=v, L=L)
            SR["runs"].append(dict(L=L, axis=an, s=float(s), lo=float(np.mean(xs)), mass=float(np.mean(ms)), ev=float(e[0])))
    for an in ("desc", "wing", "nowings"):
        for s in (-1, -0.5, 0, 0.5, 1):
            SR["free"].append(dict(L=L, axis=an, s=s, samples=sample(CTX["W"] + [U(QF)], 8, 45, v=AX[an] * s * 0.5 * NORM, L=L)))
    SR[f"cos_{L}"] = cos; log("steer", L)
json.dump(SR, open(f"{OUT}/steer_bin.json", "w"), indent=1); log("done3"); open(f"{OUT}/DONE3", "w").write("ok")
