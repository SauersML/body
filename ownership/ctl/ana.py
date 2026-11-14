"""python ana.py OUT : binding transfer (train 'wing' sentences, test wordless 'desc') per context; ownership summaries."""
import json, sys, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
import ws2 as WS
O = sys.argv[1]; KEEP = [18, 24, 27, 32]
R = [(s, p, r, u) for (s, p, r, u) in WS.spots() for k in WS.KN for i in range(8)]
side, pos, reg, surf = [np.array(x) for x in zip(*R)]; spot = side * 30 + pos * 6 + reg * 2 + surf; SP = np.arange(60)
def transfer(Xa, Xb):
    A = Xa - Xa.mean(0); B = Xb - Xb.mean(0); pca = PCA(128, random_state=0).fit(A); Ar, Br = pca.transform(A), pca.transform(B)
    sd = Ar.std(0) + 1e-6; Ar /= sd; Br /= sd; out = {}
    for k, y in (("reg", reg), ("surf", surf)): out[k] = float((RidgeClassifier(alpha=100).fit(Ar, y).predict(Br) == y).mean())
    CA = np.stack([A[spot == s].mean(0) for s in SP]); CB = np.stack([B[spot == s].mean(0) for s in SP])
    CA /= np.linalg.norm(CA, axis=1, keepdims=True); CB /= np.linalg.norm(CB, axis=1, keepdims=True)
    out["spot"] = float(((CB @ CA.T).argmax(1) == SP).mean()); return out
print("binding (train wing -> test wordless desc): region / surface / exact spot; chance 33/50/1.7")
for key in ["W_you", "N_you", "T_her", "TN_her", "W_her", "T_you"]:
    for L in (24, 27, 32):
        j = KEEP.index(L); a = np.load(f"{O}/H_{key}_wing.npy")[:, j].astype(np.float32); b = np.load(f"{O}/H_{key}_desc.npy")[:, j].astype(np.float32)
        t = transfer(a, b); print(f"  {key:7s} L{L}: {t['reg']*100:.0f} / {t['surf']*100:.0f} / {t['spot']*100:.0f}")
Rz = json.load(open(f"{O}/resO.json")); m = lambda v: round(float(np.mean(v)), 2)
print("owner axis (0 = the woman's, 1 = your)")
print("  heldout your", m(Rz["heldout_your"]), "woman's", m(Rz["heldout_woman"]), "| minimal pair 'her'", m(Rz["mp_her"]),
      "| pairs your>her", m(np.array(Rz["mp_your"]) > np.array(Rz["mp_her"])))
print("  objects your/her/the", m(Rz["obj_your"]), m(Rz["obj_her"]), m(Rz["obj_the"]), "| cos(pronoun dir, uY)", round(Rz["cos_pron_uY"], 2),
      "| pron-dir body sep / object sep", round(Rz["pron_dir_body_sep"], 2))
print("  scenes: owner, probe -> axis score, behaviour log-odds mine:hers (mean over limbs x 3 wordings); per wording")
S = Rz["scenes"]
for probe in ("the", "your", "her"):
    for own in ("you", "her"):
        rr = [r for r in S if r["probe"] == probe and r["owner"] == own]
        print(f"   scene owner={own:3s} probe='{probe} left X': axis {m([np.mean(r['score']) for r in rr])}  mine-lo {m([np.mean(r['mine_lo']) for r in rr])}  per-wording axis",
              [m([np.mean(r['score']) for r in rr if r['scene'] == s]) for s in range(3)])
