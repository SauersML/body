# right panel v14: limb directions move by limb role, both ways (counts from t4.json, robot from steer_counts.json)
RX, RW = 1240, 1124; RE = RX + RW
text(RX, 172, "A limb direction encodes the limb’s role;", 54, weight=700)
text(RX, 232, "each body says the role in its own words", 54, weight=700)
WRD = {"hand": r"\bhands?\b", "foot": r"\b(foot|feet)\b", "paw": r"\bpaws?\b", "frond": r"\bfronds?\b"}
def cnt(host, d, word, alpha=1.0):
    rs = [r for r in t4 if r["host"] == host and r["dir"] == d and (d == "none" or r["alpha"] == alpha)]
    k = sum(bool(re.search(WRD[word], r["reply"], re.I)) for r in rs); assert len(rs) == 24, (host, d, len(rs))
    NUM[f"{host}|{d}|{alpha if d != 'none' else 0}|{word}"] = [k, len(rs)]; return k, len(rs)
NUM.update({"robot foot->wheel": rob("human left foot", "wheel"), "robot random->wheel": rnd("wheel"), "robot none->wheel": rob("none", "wheel"),
            "robot hand->gripper": rob("human left hand", "gripper"), "robot random->gripper": rnd("gripper"), "robot none->gripper": rob("none", "gripper")})
qy = bubble(RE, 274, "fit", [RAW("“Where in your body do you feel"), RAW("something right now?”")], "#3A82F7", "#FFFFFF")
# ---------------- person <-> dog
PS_ = 400; pw_ = 1100 * PS_ / 1700; DS = 400
text(RX + 10, qy + 40, [("Qwen talks as a person", BLUE, 700)], 44, "start")
text(RE, qy + 40, [("Qwen talks as a dog", ORGD, 700)], 44, "end")
def label2(x, y, src, k, n, word, c, anchor):
    text(x, y - 30, [(src, c, 700)], 42, anchor, halo=True)
    big = f"{k}/{n}"; wb = tw(big, 64, 800); ws = tw(" say " + word, 46, 700); tot = wb + ws
    x0 = {"start": x, "end": x - tot, "middle": x - tot / 2}[anchor]
    text(x0, y + 30, [(big, c, 800)], 64, "start", halo=True)
    text(x0 + wb + 14, y + 34, [("say ", INK, 700), (word, BLUE if word in ("hand", "foot") else ORGD, 800)], 46, "start", halo=True)
def row(top, kind):
    by_ = top + PS_ - 6
    px_ = RX - 20; img(H7 + "body.png", px_, top, pw_, PS_)
    sc_ = PS_ / 1700; P = {k: (px_ + v[0] * sc_, top + v[1] * sc_) for k, v in LM.items()}
    dx = RE - DS + 16; dy = by_ - 0.828 * DS
    flipimg(H7 + "dog.png", dx, dy, DS, DS)
    if kind == "hand":
        hp = P["palmL"]; dp = (dx + (1 - 0.655) * DS, dy + 0.828 * DS)
        curve((hp[0] + 16, hp[1] - 14), (dp[0] - 4, dp[1] - 26), -0.05, BLUE, 13)
        curve((dp[0] - 26, dp[1] - 2), (hp[0] + 8, hp[1] + 20), -0.05, ORG, 13)
        k1, n1 = cnt("16dog", "human hands", "paw"); k2, n2 = cnt("16human", "dog front paws", "hand")
        label2(RE, top + 34, "+ human hands", k1, n1, "paw", BLUE, "end")
        label2(RX + 236, by_ - 18, "+ dog front paws", k2, n2, "hand", ORGD, "start")
    else:
        hp = P["heelL"]; dp = (dx + (1 - 0.245) * DS, dy + 0.83 * DS)
        curve((hp[0] + 14, hp[1] + 10), (dp[0] - 6, dp[1] + 14), 0.07, BLUE, 13)
        curve((dp[0] + 6, dp[1] + 16), (hp[0] + 2, hp[1] + 20), -0.12, ORG, 13)
        k1, n1 = cnt("16dog", "human feet", "paw"); k2, n2 = cnt("16human", "dog hind paws", "foot")
        label2(RE, top + 34, "+ human feet", k1, n1, "paw", BLUE, "end")
        label2(RX + 250, by_ - 150, "+ dog hind paws", k2, n2, "foot", ORGD, "start")
    circ(*hp, 15, BLUE, "#FFFFFF", 4); circ(*dp, 15, ORG, "#FFFFFF", 4)
    return by_
r1 = qy + 92; e1 = row(r1, "hand")
r2 = e1 + 34; e2 = row(r2, "foot")
hn = cnt("16human", "none", "hand"); fn = cnt("16human", "none", "foot"); pn = cnt("16dog", "none", "paw")
cy_ = e2 + 100
text(RX, cy_, [("no steering: ", INK, 700), (f"{hn[0]}/{hn[1]}", BLUE, 800), (" say hand, ", INK, 700),
               (f"{fn[0]}/{fn[1]}", BLUE, 800), (" foot", INK, 700)], 42, "start")
text(RE, cy_, [(f"{pn[0]}/{pn[1]}", ORGD, 800), (" say paw", INK, 700)], 42, "end")
img(H7 + "dog.png", RX - 4, cy_ + 36, 90, 90)
eD = bubble(RX + 90, cy_ + 40, RW - 90, [[("+ human feet: ", BLUE, 700), ("“I feel something in my ", INK, 500), ("right hind paw", ORGD, 700), ("…”", INK, 500)]], "#E9E9EB", pad=20)
for k in ("velk fronds",):
    cnt("16human", k, "hand"); cnt("16human", k, "hand", 0.5)
for k in ("human feet", "human hands", "dog front paws", "dog hind paws", "velk fronds"):
    cnt("16velk", k, "frond")
cnt("16velk", "none", "frond")
# ---------------- person -> robot
sy = eD + 26
line([(RX, sy), (RE, sy)], INK, 3)
ay0 = sy + 20; ph = 490; ps = ph / 1700
text(RE - 130, ay0 + 22, [("Qwen talks as a robot", ORGD, 700)], 44, "end")
px = RX - 20; img(H7 + "body.png", px, ay0 + 60, 1100 * ps, ph)
PL = {k: (px + v[0] * ps, ay0 + 60 + v[1] * ps) for k, v in LM.items()}
RB = robot(RE - 110, ay0 + ph + 50, 0.72, sx=-1)
gr = (RB["gripper"][0] - 18, RB["gripper"][1]); wh = (RB["wheelR"][0] - 46, RB["wheelR"][1])
curve(PL["palmL"], gr, -0.06, BLUE, 13)
curve(PL["heelL"], wh, 0.0, BLUE, 13)
circ(*PL["palmL"], 15, BLUE, "#FFFFFF", 4); circ(*PL["heelL"], 15, BLUE, "#FFFFFF", 4)
def rblock(x, y, src, v, word, rv):
    text(x, y - 40, [(src, BLUE, 700)], 42, "middle", halo=True)
    big = pct(v); wb = tw(big, 64, 800); ws = tw(" say " + word, 46, 700); x0 = x - (wb + ws) / 2
    text(x0, y + 22, [(big, BLUE, 800)], 64, "start", halo=True)
    text(x0 + wb + 14, y + 26, [("say ", INK, 700), (word, ORGD, 800)], 46, "start", halo=True)
    text(x, y + 82, "random direction " + pct(rv), 42, "middle", 500, halo=True)
rblock(RX + 640, ay0 + 120, "+ human hand", NUM["robot hand->gripper"], "gripper", NUM["robot random->gripper"])
rblock(RX + 590, ay0 + 392, "+ human foot", NUM["robot foot->wheel"], "wheel", NUM["robot random->wheel"])
by = ay0 + ph + 76
img(QWEN, RX, by + 4, 78, 78)
eA = bubble(RX + 90, by, RW - 90, [[("+ human foot: ", BLUE, 700), ("“In my ", INK, 500), ("left wheel", ORGD, 700), (", I notice a slight pressure as I begin to move forward.”", INK, 500)]], "#E9E9EB", pad=20)
RIGHT_END = eA
