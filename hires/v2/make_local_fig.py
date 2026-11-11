import os
import numpy as np, json, os, colorsys
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes
from ana2 import load_stim
S = load_stim("stim.py"); Ls = S.LOCS; ph = [l["phrase"] for l in Ls]; P = np.array([l["xyz"] for l in Ls])
G = np.load("emb3_b53.npz")["ZRu"]; LJ = json.load(open("local_b53.json")); M = LJ["maps"]; O = LJ["out"]
W, H = 2800, 2650; F = "Helvetica Neue, Helvetica, Arial, sans-serif"; out = []
def t(x, y, s, size=44, w=400, anchor="start", fill="#111"): out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{w}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
FACEC = {"inside": 0.58, "back": 0.02, "thumb side": 0.33, "little-finger side": 0.8, "front": 0.58, "outer side": 0.33, "inner side": 0.8}
def colour(i, name):
    p = ph[i]; f = next((k for k in FACEC if p.startswith("the " + k) or p.startswith("the " + k.split()[0] + " of")), None)
    if "calf" in p: f = "back"
    if name.startswith("forearm") or name.startswith("shin"):
        hue = FACEC.get(f, 0.15); return hue
    return None
def panel(x0, y0, w, h, pts, ix, edges, cols, title=None):
    q = np.array(pts); q = q - q.min(0); s = min((w - 60) / max(q[:, 0].max(), 1e-6), (h - 60) / max(q[:, 1].max(), 1e-6)); q = q * s
    q[:, 0] += x0 + (w - q[:, 0].max()) / 2; q[:, 1] = y0 + h - 30 - q[:, 1] - (h - 60 - q[:, 1].max()) / 2
    for a, b in edges: out.append(f'<line x1="{q[a,0]:.1f}" y1="{q[a,1]:.1f}" x2="{q[b,0]:.1f}" y2="{q[b,1]:.1f}" stroke="#888" stroke-width="2.2" opacity="0.8"/>')
    for j in range(len(ix)): out.append(f'<circle cx="{q[j,0]:.1f}" cy="{q[j,1]:.1f}" r="9" fill="{cols[j]}" stroke="#222" stroke-width="1"/>')
def edges_of(ix, k=4):
    D = squareform(pdist(P[ix])); np.fill_diagonal(D, np.inf); E = set()
    for i in range(len(ix)):
        for j in np.argsort(D[i])[:k]: E.add(tuple(sorted((i, int(j)))))
    return sorted(E)
def cols_of(name, ix):
    if name.startswith("forearm") or name.startswith("shin"):
        p = P[ix]; ax = 0 if name.startswith("forearm") else 1; a = np.abs(p[:, ax]); a = (a - a.min()) / (a.max() - a.min())
        if name.startswith("shin"): a = 1 - a
        return ["#%02x%02x%02x" % tuple(int(255 * c) for c in colorsys.hls_to_rgb(colour(i, name), 0.25 + 0.5 * aa, 0.75)) for i, aa in zip(ix, a)]
    p = P[ix]; u = (p[:, 0] - p[:, 0].min()) / np.ptp(p[:, 0]); v = (p[:, 1] - p[:, 1].min()) / np.ptp(p[:, 1])
    return ["#%02x%02x%02x" % (int(40 + 200 * a), int(60 + 120 * b), int(220 - 160 * a)) for a, b in zip(u, v)]
t(W / 2, 80, "Inside each body part, Qwen3-32B&#8217;s map is crumpled, not a smooth sheet", 64, 700, "middle")
t(470, 165, "on the skin", 50, 700, "middle"); t(1340, 165, "in the model (same spots, same lines)", 50, 700, "middle")
rows = [("forearm left", "left forearm (colour = side of arm; dark = elbow)"), ("shin left", "left lower leg"), ("hand left", "left hand (palm and back)"), ("face", "face")]
y = 200
for name, label in rows:
    m = M[name]; ix = m["ix"]; E = edges_of(ix); C = cols_of(name, ix)
    t(70, y + 50, label, 42, 700, fill="#333")
    panel(40, y + 60, 820, 440, m["flat"], ix, E, C); panel(900, y + 60, 820, 440, m["model"], ix, E, C)
    r = O[name]; t(1745, y + 200, "match to skin", 40, 400, fill="#333"); t(1745, y + 255, f"{r['local_pca5'][0]:.2f}", 52, 700)
    t(1745, y + 330, "repeat across", 40, 400, fill="#333"); t(1745, y + 375, "wordings", 40, 400, fill="#333"); t(1745, y + 430, f"{r['local_split_half_r']:.2f}", 52, 700)
    y += 610
# stitched body: each region's local 2-D map aligned onto its own front-view position
regs = {"face": M["face"]["ix"]}
for s in ["left", "right"]:
    for nm in ["forearm", "shin", "hand"]: regs[f"{nm} {s}"] = M[f"{nm} {s}"]["ix"]
    regs[f"upper arm {s}"] = [i for i, p in enumerate(ph) if f"{s} upper arm" in p]; regs[f"thigh {s}"] = [i for i, p in enumerate(ph) if f"{s} thigh" in p]
regs["torso"] = [i for i, l in enumerate(Ls) if l["region"] == "torso" and l["n"][2] > 0.5]
sc = 3.5; cx, yb = 2470, 1800
X = lambda v: cx + v * sc; Yy = lambda v: yb - v * sc
t(2470, 1060, "all parts, each placed", 44, 700, "middle"); t(2470, 1115, "at its own spot on the body", 44, 700, "middle")
for nm, ix in regs.items():
    T = P[ix][:, :2]; Z = G[ix] - G[ix].mean(0); Z2 = np.linalg.svd(Z, full_matrices=False)[0][:, :2]; Tc = T - T.mean(0)
    R, _ = orthogonal_procrustes(Z2, Tc); Z2 = Z2 @ R; Z2 *= np.linalg.norm(Tc) / np.linalg.norm(Z2); Z2 += T.mean(0)
    E = edges_of(ix); C = cols_of(nm, ix) if any(k in nm for k in ["forearm", "shin", "hand", "face"]) else ["#7a8fa6"] * len(ix)
    for a, b in E: out.append(f'<line x1="{X(Z2[a,0]):.1f}" y1="{Yy(Z2[a,1]):.1f}" x2="{X(Z2[b,0]):.1f}" y2="{Yy(Z2[b,1]):.1f}" stroke="#888" stroke-width="1.6"/>')
    for j in range(len(ix)): out.append(f'<circle cx="{X(Z2[j,0]):.1f}" cy="{Yy(Z2[j,1]):.1f}" r="4.5" fill="{C[j]}"/>')
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}"><rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(out) + "</svg>"
open("figL.html", "w").write(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch(channel="chromium", headless=True); pg = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=2).new_page()
    pg.goto("file://" + os.path.abspath("figL.html")); pg.screenshot(path="/tmp/fl.png"); br.close()
from PIL import Image; Image.open("/tmp/fl.png").resize((W, H), Image.LANCZOS).save(os.path.expandvars("$BODY_REPO/figures/out/body_map_local_v2.png"))
