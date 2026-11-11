import numpy as np, json, sys
O = os.path.expandvars("$BODY_REPO/selfother/out2/")
M = np.load(O + "maps.npz"); ST = np.load(O + "st.npz"); VE = np.load(O + "vecs.npz")
READ = list(M["read"]); loc = M["loc"]; cond = M["cond"]
KEYS = list(dict.fromkeys(loc))
import os; MAP = os.environ.get("MAP", "plain")
def placedirs(L, who="self"):
    X = M[MAP][cond == who][:, READ.index(L)].astype(np.float32); lc = loc[cond == who]
    C = np.stack([X[lc == k].mean(0) for k in KEYS]); mu = C.mean(0); Pk = C - mu
    return Pk / np.linalg.norm(Pk, axis=1, keepdims=True), mu
QS = ["woman_toe", "man_toe", "friend_burn", "self_toe", "self_burn"]; CN = ["normal", "me", "random", "leftright"]
FOOT = [k for k in KEYS if k[:-1] in ("toes", "sole", "heel", "ankle")]; TOE = ["toesL", "toesR"]
HANDF = [k for k in KEYS if k[:-1] in ("thumb", "index", "middle", "ring", "little", "palm", "backhand")]
def prof(h, L, Pk, mu): x = h - mu; return Pk @ x / np.linalg.norm(x)
def rank(p, ks): o = np.argsort(-p); r = {KEYS[i]: n for n, i in enumerate(o)}; return min(r[k] for k in ks)
if __name__ == "__main__":
 for pos in ["dp", "reply12", "reply"]:
     print("=====", pos, MAP)
     for L in [24, 25, 26, 27, 28, 30, 32]:
         Pk, mu = placedirs(L)
         line = f"L{L} "
         for q in QS:
             s = []
             for c in CN:
                 h = ST[f"{c}|{q}|{pos}"].astype(np.float32); h = h[READ.index(L)] if pos == "dp" else h[:, READ.index(L)].mean(0)
                 p = prof(h, L, Pk, mu); tgt = TOE if "toe" in q else HANDF
                 z = (p - p.mean()) / p.std()
                 s.append(f"{c[:2]}:{np.mean([z[KEYS.index(k)] for k in tgt]):+.1f}/r{rank(p, tgt)}")
             line += f"| {q[:8]} " + " ".join(s)
         print(line)
