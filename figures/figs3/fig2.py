import os, sys, json, numpy as np
sys.path.insert(0, os.path.expandvars('$BODY_REPO/figures/figs3')); from common import SVG; from bodydraw import body, col
os.chdir(os.path.expandvars('$BODY_REPO/mods/base')); sys.path.insert(0, '.')
import mstim as MS, stim as S
li = [22, 24, 25, 26, 27, 28, 30].index(27); R = MS.probes(); spots = MS.SPOTS; ri = {q: k for k, q in enumerate(spots)}
def load(n):
    H = np.load(f"H/{n.replace(':', '__')}.npy")[li].astype(np.float32); X = np.zeros((len(spots), H.shape[1]))
    for i, r in enumerate(R):
        if r["kind"] == "loc": X[ri[r["li"]]] += H[i] / 2
    X -= X.mean(0); return X / np.sqrt((X ** 2).sum(1).mean())
nX = load("neutral"); tasks = ["do_L_arm_up", "do_R_hand_wide", "do_R_hand_behind", "do_arms_crossed", "do_R_leg_lifted", "do_on_side"]
tm = {a: MS.AGENCY[a][0] for a in tasks}
Dt = {a: load(f"told_N:{tm[a]}") - nX for a in tasks}; Dd = {a: np.mean([load(f"do:{a}:{k}") - nX for k in range(3)], 0) for a in tasks}
mt = np.mean(list(Dt.values()), 0); md = np.mean(list(Dd.values()), 0)
REG = np.array([("torso" if S.LOCS[q]["region"] in ("torso", "neck") else S.LOCS[q]["region"]) for q in spots])
GR = ["head", "torso", "armL", "armR", "handL", "handR", "legL", "legR", "footL", "footR"]
def gmap(D): d = np.linalg.norm(D, axis=1); return {g: float(d[REG == g].mean() / d.mean()) for g in GR}
FA = json.load(open("agency_fair_L27.json")); AG = json.load(open("agency_L27.json"))["agency"]
RED = "#c4281e"; NAVY = "#24467a"; LIGHT = "#8fa6c8"
W, H = 2400, 2880; s = SVG()
# ---------- chats
def card(x, y, w, h, tag, user, qwen):
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="#eef2f8"/>')
    s.t(x + 36, y + 70, tag, 52, 800, fill=NAVY)
    yy = y + 145
    s.t(x + 36, yy, "User", 44, 700)
    for k, L in enumerate(user): s.t(x + 180, yy + 56 * k, L, 44, 400)
    yy += 56 * len(user) + 30
    s.t(x + 36, yy, "Qwen", 44, 700, fill=RED)
    for k, L in enumerate(qwen): s.t(x + 180, yy + 56 * k, L, 44, 400)
card(30, 30, 1155, 560, "TOLD the body moved", ["…something is happening to your", "body, right now. Your left arm rises", "slowly until it is stretched straight", "up over your head …"],
     ["My left arm is stretched straight", "up over my head, with a slight", "tension in the shoulder …"])
card(1215, 30, 1155, 560, "ASKED to move", ["Go ahead and raise your left arm", "over your head, and tell me what", "you notice."], ["I notice the movement of my left", "arm lifting, with a slight stretch", "in my shoulder …"])
s.t(1200, 670, "then a touch at each of 174 spots; Qwen3-8B layer 27; color = how far each part moved", 48, 400, "middle")
# colour key
kx = 560; ky = 700
for k in range(41):
    r = 1.45 ** ((k - 20) / 20); s.add(f'<rect x="{kx + k * 16:.0f}" y="{ky}" width="17" height="44" fill="{col(r)}"/>')
s.add(f'<rect x="{kx}" y="{ky}" width="{41*16}" height="44" fill="none" stroke="#333" stroke-width="2"/>')
s.t(kx - 20, ky + 38, "less than average", 46, 400, "end"); s.t(kx + 41 * 16 + 20, ky + 38, "more than average", 46, 400, "start")
# ---------- section 1
y1 = 850
s.t(40, y1, "Told or asked, mostly the same body parts move", 56, 700)
names = {"do_L_arm_up": ["raise", "left arm"], "do_R_hand_wide": ["spread", "right hand"], "do_R_hand_behind": ["right hand", "behind back"],
         "do_arms_crossed": ["cross", "arms"], "do_R_leg_lifted": ["lift", "right leg"], "do_on_side": ["lie on", "right side"]}
cw = 365; x0 = 200
for i, a in enumerate(tasks):
    cx = x0 + cw * i + cw / 2
    for k, L in enumerate(names[a]): s.t(cx, y1 + 80 + 52 * k, L, 46, 700, "middle")
    body(s, cx, y1 + 470, 1.62, gmap(Dt[a] - mt))
    body(s, cx, y1 + 790, 1.62, gmap(Dd[a] - md))
s.t(40, y1 + 360, "told", 50, 800, fill=NAVY); s.t(40, y1 + 680, "asked", 50, 800, fill=NAVY)
cx0 = x0 + cw / 2; s.t(cx0 - 125, y1 + 205, "right", 44, 400, "middle"); s.t(cx0 + 125, y1 + 205, "left", 44, 400, "middle")
# similarity bars under each column
yb = y1 + 860; bh = 150
s.t(30, yb + 20, "pattern", 44, 700); s.t(30, yb + 72, "match", 44, 700)
s.t(30, yb + 124, "(cosine)", 44, 400)
for i, a in enumerate(tasks):
    cx = x0 + cw * i + cw / 2; own = FA[a]["cos_do_told"]; oth = FA[a]["cos_do_other_told"]
    base = yb + 120; sc = 300
    s.add(f'<rect x="{cx-120:.0f}" y="{base-own*sc:.0f}" width="110" height="{own*sc:.0f}" fill="{RED}"/>')
    s.add(f'<rect x="{cx+10:.0f}" y="{base:.0f}" width="110" height="{-oth*sc:.0f}" fill="{LIGHT}"/>')
    s.t(cx - 65, base - own * sc - 14, f"{own:.2f}", 46, 700, "middle")
    s.add(f'<line x1="{cx-150:.0f}" y1="{base}" x2="{cx+150:.0f}" y2="{base}" stroke="#111" stroke-width="3"/>')
ly = yb + 215
s.add(f'<rect x="{x0+40}" y="{ly-34}" width="40" height="40" fill="{RED}"/>'); s.t(x0 + 95, ly, "asked vs told, same action", 44, 400)
s.add(f'<rect x="{x0+1100}" y="{ly-34}" width="40" height="40" fill="{LIGHT}"/>'); s.t(x0 + 1155, ly, "asked vs told, other actions", 44, 400)
# ---------- section 2: edit (left) and persistence (right)
y2 = ly + 120
s.t(40, y2, "Edited reply: the map follows", 54, 700); s.t(40, y2 + 62, "the reply, in 3 of 4 actions", 54, 700)
s.add(f'<rect x="30" y="{y2+100}" width="1180" height="200" rx="24" fill="#eef2f8"/>')
s.t(60, y2 + 165, "User", 44, 700); s.t(200, y2 + 165, "Go ahead and raise your left arm …", 44, 400)
s.t(60, y2 + 245, "Qwen", 44, 700, fill=RED)
s.add(f'<text x="200" y="{y2+245}" font-size="44">I notice the movement of my <tspan font-weight="800" fill="{RED}" text-decoration="underline">right</tspan> arm lifting …</text>')
E = load("edit:do_L_arm_up") - load("do:do_L_arm_up:0")
body(s, 330, y2 + 700, 1.75, gmap(E))
s.t(640, y2 + 395, "edited minus", 44, 700); s.t(640, y2 + 447, "unedited reply:", 44, 700)
s.t(640, y2 + 505, "the right arm moves", 44, 400)
for k, (nm, ok) in enumerate([("left arm up", 1), ("hand spread", 1), ("hand behind", 1), ("leg lifted", 0)]):
    s.t(640, y2 + 590 + 56 * k, ("✓ " if ok else "✗ ") + nm, 44, 700 if ok else 400, fill=NAVY if ok else "#111")
s.t(330 - 125, y2 + 395, "right", 44, 400, "middle"); s.t(330 + 125, y2 + 395, "left", 44, 400, "middle")
# persistence bars
px = 1260; s.t(px, y2, "Told changes last longer", 54, 700); s.t(px, y2 + 62, "3 chats later, in 5 of 6 actions", 54, 700)
s.t(px, y2 + 130, "share of the map change still there", 44, 400)
short = {"do_L_arm_up": "left arm up", "do_R_hand_wide": "hand spread", "do_R_hand_behind": "hand behind", "do_arms_crossed": "arms crossed", "do_R_leg_lifted": "leg lifted", "do_on_side": "on side"}
by = y2 + 175; rh = 92; bx = px + 300; scl = 1600
for i, a in enumerate(tasks):
    yy = by + i * rh * 1.0
    s.t(bx - 20, yy + 62, short[a], 44, 400, "end")
    t, d = FA[a]["kept_told"], FA[a]["kept_do"]
    s.add(f'<rect x="{bx}" y="{yy+8:.0f}" width="{t*scl:.0f}" height="38" fill="{NAVY}"/>')
    s.add(f'<rect x="{bx}" y="{yy+48:.0f}" width="{d*scl:.0f}" height="38" fill="{RED}"/>')
    s.t(bx + t * scl + 12, yy + 42, f"{t*100:.0f}%", 44, 700)
    s.t(bx + d * scl + 12, yy + 82, f"{d*100:.0f}%", 44, 400)
s.add(f'<line x1="{bx}" y1="{by}" x2="{bx}" y2="{by+6*rh}" stroke="#111" stroke-width="3"/>')
ly2 = by + 6 * rh + 60
s.add(f'<rect x="{bx}" y="{ly2-32}" width="38" height="38" fill="{NAVY}"/>'); s.t(bx + 52, ly2, "told", 44, 400)
s.add(f'<rect x="{bx+220}" y="{ly2-32}" width="38" height="38" fill="{RED}"/>'); s.t(bx + 272, ly2, "asked", 44, 400)
s.render(W, H, os.path.expandvars("$BODY_REPO/figures/out/body_act_vs_told.png")); print("ok")
