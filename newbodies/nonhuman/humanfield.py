"""Smooth color field over the real human mesh (MakeHuman neutral, CC0): each of the 54 places gets the field's color at its landmark.
Renders front and back views as PNG (transparent), and writes place colors + 2-D landmark positions to humanfield.json."""
import numpy as np, json, sys, os
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stimuli as S
CF = os.path.expandvars("$BODY_WORK/")
V, F = [], []
for l in open(CF + "fig/human/neutral.obj"):
    if l.startswith("v "): V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:]])
V = np.array(V); F = np.array(F)
def hx(c): c = c.lstrip("#"); return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float)
# anchors (x, y, z, color); +x = the body's left
A = [((0, 8.2, 0.6), "#B8322A"), ((0, 6.9, 1.5), "#D94F3A"), ((0.8, 7.0, 0.3), "#E0607E"), ((-0.8, 7.0, 0.3), "#E0607E"), ((0, 6.0, 0.4), "#EE8A3C"),
     ((0, 3.9, 1.7), "#E8B33F"), ((0, 1.6, 1.9), "#C7B94A"), ((0, 4.0, -0.6), "#9A6A3A"), ((0, 1.4, -0.5), "#6B4A2E")]
for s in (1, -1):
    A += [((s * 1.9, 5.1, 0.5), "#A9CBEA"), ((s * 3.0, 3.6, 0.9), "#6FA3D8"), ((s * 3.9, 2.4, 1.6), "#3F78C0"), ((s * 4.7, 1.1, 2.4), "#1D4A96"),
          ((s * 1.1, 0.3, 0.8), "#B6DDA6"), ((s * 1.5, -3.4, 1.2), "#5FAE73"), ((s * 1.9, -6.9, 1.2), "#2E8075"), ((s * 2.0, -8.0, 2.4), "#1C5560")]
AP = np.array([a for a, _ in A], float); AC = np.array([hx(c) for _, c in A])
def field(P, sig=0.9):
    d2 = ((P[:, None, :] - AP[None]) ** 2).sum(-1); w = np.exp(-d2 / (2 * sig ** 2)) + 1e-12
    return (w[:, :, None] * AC[None]).sum(1) / w.sum(1, keepdims=True)
# landmarks for the 54 places (approximate, then snapped to the nearest surface vertex)
LM = {"forehead": (0, 7.6, 1.6), "crown": (0, 8.4, 0.6), "cheekL": (0.55, 6.9, 1.4), "cheekR": (-0.55, 6.9, 1.4), "lips": (0, 6.6, 1.6), "tongue": (0, 6.6, 1.3),
      "earL": (0.82, 7.0, 0.3), "earR": (-0.82, 7.0, 0.3), "neckF": (0, 5.9, 1.1), "neckB": (0, 6.0, -0.2), "chest": (0, 4.2, 1.9), "belly": (0, 1.8, 2.0),
      "upback": (0, 4.2, -0.6), "lowback": (0, 1.5, -0.5)}
ARMP = {"shoulder": (1.8, 5.2, 0.5), "upperarm": (2.4, 4.4, 0.6), "elbow": (3.0, 3.6, 0.9), "forearm": (3.5, 3.0, 1.2), "wrist": (4.0, 2.2, 1.7),
        "palm": (4.4, 1.7, 2.2), "backhand": (4.5, 1.8, 2.0), "thumb": (4.3, 1.3, 2.6), "index": (4.6, 1.0, 2.7), "middle": (4.8, 1.0, 2.6),
        "ring": (4.9, 1.1, 2.5), "little": (4.95, 1.25, 2.4)}
LEGP = {"hip": (1.3, 0.6, 0.6), "thigh": (1.2, -1.5, 1.0), "knee": (1.4, -3.5, 1.3), "shin": (1.6, -5.2, 1.2), "ankle": (1.9, -7.2, 0.8),
        "heel": (1.9, -8.0, 0.0), "sole": (2.0, -8.1, 1.4), "toes": (2.1, -8.0, 2.6)}
for k, p in {**ARMP, **LEGP}.items():
    LM[k + "L"] = p; LM[k + "R"] = (-p[0], p[1], p[2])
used = np.unique(F.ravel()); U = V[used]
snap = {k: U[((U - np.array(p)) ** 2).sum(1).argmin()] for k, p in LM.items()}
keys = [l["key"] for l in S.LOCS]
PC = {k: field(np.array([snap[k]]))[0] for k in keys}
C = field(V)
def render(view, H=1600, pad=20):
    X = V.copy()
    if view == "back": X[:, 0] *= -1; X[:, 2] *= -1
    lo, hi = U.min(0), U.max(0); sc = (H - 2 * pad) / (hi[1] - lo[1]); Wd = int((hi[0] - lo[0]) * sc + 2 * pad)
    P2 = np.stack([(X[:, 0] + hi[0]) * sc + pad, (hi[1] - X[:, 1]) * sc + pad], 1)
    img = Image.new("RGBA", (Wd * 2, H * 2), (255, 255, 255, 0)); dr = ImageDraw.Draw(img)
    fz = X[F][:, :, 2].mean(1); order = np.argsort(fz)
    a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]; n = np.cross(b - a, c - a); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    L = np.array([-0.35, 0.45, 0.82]); L /= np.linalg.norm(L); sh = np.clip(n @ L, 0, 1)
    for i in order:
        if n[i, 2] < -0.05: continue
        col = C[F[i]].mean(0) * (0.62 + 0.38 * sh[i]) + 255 * 0.06
        dr.polygon([tuple(2 * P2[j]) for j in F[i]], fill=tuple(int(min(255, v)) for v in col) + (255,))
    img = img.resize((Wd, H), Image.LANCZOS); img.save(CF + f"body/nonhuman/human_{view}.png")
    pos = {k: [float((snap[k][0] * (-1 if view == "back" else 1) + hi[0]) * sc + pad), float((hi[1] - snap[k][1]) * sc + pad)] for k in keys}
    return dict(w=Wd, h=H, pos=pos)
out = dict(colors={k: [int(x) for x in v] for k, v in PC.items()}, front=render("front"), back=render("back"))
json.dump(out, open(CF + "body/nonhuman/humanfield.json", "w")); print("ok", out["front"]["w"], out["front"]["h"])
