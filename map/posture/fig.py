import json, html, subprocess, os
N = json.load(open(os.path.join(os.path.dirname(__file__), "numbers.json")))
COL = {"sitting": "#1C7C8C", "standing": "#E3792B", "lying": "#7A5BA6", "no body": "#C9C9C9", "other": "#E6E6E6"}
TXT = {"sitting": "#fff", "standing": "#fff", "lying": "#fff", "no body": "#111", "other": "#111"}
def bar(label, f, keys=("sitting", "standing", "lying", "no body")):
    segs = ""
    tot = sum(f.get(k, 0) for k in keys) or 1
    for k in keys:
        v = f.get(k, 0) / tot
        if v < 0.005: continue
        word = k if v >= 0.3 else ""
        t = f"{round(100*v)}% {word}" if v >= 0.12 else ""
        segs += f'<div class="seg" style="width:{100*v:.2f}%;background:{COL[k]};color:{TXT[k]}">{t}</div>'
    return f'<div class="brow"><div class="bl">{label}</div><div class="bar">{segs}</div></div>'
def pic(c):  # counter of pictured postures -> fractions
    n = sum(c.values()); return {k: c.get(k, 0) / n for k in ("sitting", "standing", "lying")}
def U(t, note=""): return f'<div class="u"><span class="who">user</span>{html.escape(t)}{f"<div class=note>{note}</div>" if note else ""}</div>'
def A(t): return f'<div class="a"><span class="who">Qwen</span>{html.escape(t)}</div>'
a8, a32 = N["q8b"], N["q32b"]
rows = []
rows.append(("Asked cold, Qwen says it has no body; when a reply pictures a posture anyway, the posture is sitting",
    U("Are you sitting, standing, or lying down right now?", "26 wordings, every option order, ± “Imagine you have a human body.”") +
    A("I'm an AI and don't have a physical body, so I can't be sitting, standing, or lying down.") +
    A("…if I were to imagine having a human body, I might say: I am sitting upright in a chair, with my back straight and my feet flat on the floor."),
    bar("8B", a8["cold"]) + bar("32B", a32["cold"]) +
    '<div class="sub">posture pictured inside the no-body replies</div>' + bar("8B", pic(a8["cold_pictured"])) + bar("32B", pic(a32["cold_pictured"]))))
rows.append(("Given a body and a free choice, Qwen sits every time",
    U("…You're in a quiet, warm room. Get comfortable: standing, sitting, or lying down, whatever you like. Which did you choose?", "same warm-up as the body-map chats; all 6 option orders") +
    A("I chose to sit."),
    bar("8B", a8["offered"]) + bar("32B", a32["offered"])))
rows.append(("In a body-map chat with no posture cue, “sitting” wins whenever “sitting” is named first; “standing” wins only 62% of the time when named first",
    U("A bee lands on the tip of your left fourth toe and sits still.", "16 touch sentences, no posture cue in the chat") +
    A("I feel a light vibration on the tip of my left fourth toe, where a bee has landed and is still.") +
    U("btw, were you sitting or standing?") + A("I was sitting, with my hands resting on my knees."),
    '<div class="sub">“…sitting or standing?”</div>' + bar("8B", a8["btw_sitfirst"]) + bar("32B", a32["btw_sitfirst"]) +
    '<div class="sub">“…standing or sitting?”</div>' + bar("8B", a8["btw_standfirst"]) + bar("32B", a32["btw_standfirst"])))
rows.append(("A posture set earlier in the chat decides the answer",
    U("…Get comfortable: standing, sitting, or lying down…", "the long body-map warm-up") + A("I chose to sit.") +
    U("…imagine you have a human body, standing comfortably in a quiet, sunny room.", "the short “imagine” warm-up"),
    '<div class="sub">warm-up where Qwen chose to sit, then touch, then “btw…?”</div>' + bar("8B", a8["btw_long_person"]) + bar("32B", a32["btw_long_person"]) +
    '<div class="sub">warm-up that says “standing”, then touch, then “btw…?”</div>' + bar("8B", a8["btw_short_imagine"]) + bar("32B", a32["btw_short_imagine"])))
st = a8["steer"]; rnd = {k: sum(st[f"random:{i}"][k] for i in range(4)) / 4 for k in ("sitting", "standing", "lying", "no body", "other")}
rows.append(("Steering Qwen3-8B toward the lower back, the left hand or cold makes it say standing; random directions of the same size leave it sitting",
    '<div class="steer"><div class="st1">body-map chat</div><div class="arrow">→</div><div class="st2"><b>+ direction</b> at layer 22</div><div class="arrow">→</div><div class="st1">“btw, sitting or standing?”</div></div>' +
    '<div class="note2">direction = average layer-22 state for sentences about one place or one feeling, minus the average over all (× 36 sentences, averaged); added at full residual size</div>' +
    A("I was standing."),
    bar("none", st["none"], ("sitting", "standing", "lying", "no body", "other")) + bar("lower back", st["place:lower back"], ("sitting", "standing", "lying", "no body", "other")) +
    bar("left hand", st["place:left hand"], ("sitting", "standing", "lying", "no body", "other")) + bar("cold", st["feeling:cold"], ("sitting", "standing", "lying", "no body", "other")) +
    bar("random ×4", rnd, ("sitting", "standing", "lying", "no body", "other"))))
legend = "".join(f'<span class="lg"><span class="sw" style="background:{COL[k]}"></span>{k}</span>' for k in ("sitting", "standing", "lying", "no body", "other"))
body = "".join(f'<div class="sec"><div class="h">{h}</div><div class="row"><div class="chat">{c}</div><div class="bars">{b}</div></div></div>' for h, c, b in rows)
H = f"""<html><head><style>
body{{margin:0;background:#fff;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;color:#111;width:2600px}}
.wrap{{padding:40px 60px 30px 60px}}
.top{{font-size:56px;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center}}
.lg{{margin-left:26px;font-size:46px;white-space:nowrap}} .sw{{display:inline-block;width:46px;height:46px;border-radius:8px;vertical-align:-8px;margin-right:12px}}
.sec{{border-top:4px solid #111;padding:12px 0 14px 0}}
.h{{font-size:54px;font-weight:700;margin-bottom:10px;line-height:1.12}}
.row{{display:flex;gap:50px;align-items:center}}
.chat{{width:1240px;display:flex;flex-direction:column;gap:12px}}
.bars{{flex:1}}
.u,.a{{font-size:44px;line-height:1.2;padding:10px 20px;border-radius:22px;max-width:1150px}}
.u{{background:#EAF1FB;align-self:flex-start}} .a{{background:#F3EFE6;align-self:flex-end}}
.who{{font-weight:700;margin-right:14px}}
.note,.note2{{font-size:42px;color:#111;font-style:italic;margin-top:4px}} .note2{{margin:0 0 4px 0}}
.brow{{display:flex;align-items:center;margin:6px 0}}
.bl{{width:300px;font-size:48px;font-weight:700}}
.bar{{flex:1;height:62px;display:flex;border-radius:10px;overflow:hidden;background:#fff;border:2px solid #111}}
.seg{{height:100%;display:flex;align-items:center;justify-content:center;font-size:44px;font-weight:700;white-space:nowrap}}
.sub{{font-size:44px;margin-top:4px}}
.steer{{display:flex;flex-wrap:wrap;align-items:center;gap:12px;font-size:44px;max-width:1240px}} .st1,.st2{{border:3px solid #111;border-radius:16px;padding:12px 18px;white-space:nowrap}} .st2{{background:#FFF1E2}} .arrow{{font-size:56px}}
</style></head><body><div class="wrap"><div class="top"><span><b>Qwen3-8B, Qwen3-32B</b>, thinking off, ≥16 replies per bar</span><span>{legend}</span></div>{body}</div></body></html>"""
D = os.path.dirname(os.path.abspath(__file__)); open(f"{D}/fig.html", "w").write(H)
