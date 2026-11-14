"""Controls for the two body findings (Qwen3-8B). python ctl.py OUT
Part W: wordless-touch binding in embodied 'you' wings chat vs third-person story chat ('her' wings) vs no-wings chats.
Part O: owner axis (your - the woman's, embodiment chat, L24): minimal pairs with 'her', pronoun-free ownership from the scene, non-body your/her."""
import os, sys, json, re, copy, time, itertools, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ws2 as WS, stimuli as S
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); t0 = time.time(); DEV = "cuda"
M = "Qwen/Qwen3-8B"; KEEP = [18, 24, 27, 32]
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
U = lambda c: {"role": "user", "content": c}; A = lambda c: {"role": "assistant", "content": c}
def log(*a): print(round(time.time() - t0), *a, flush=True)
def chat(msgs, gen=True): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
def trim(c):
    c = c.strip()
    if not re.search(r"[.!?)*]\s*$", c): c = c[:max(c.rfind(". "), c.rfind("! ")) + 1] or c
    return c.strip()
def sample(msgs, n=1, mx=80, seed=0):
    ids = tok([chat(msgs)] * n, return_tensors="pt").to(DEV); torch.manual_seed(seed)
    with torch.no_grad(): g = model.generate(**ids, max_new_tokens=mx, do_sample=True, temperature=0.7, top_p=0.9)
    return [trim(tok.decode(x[ids.input_ids.shape[1]:], skip_special_tokens=True)) for x in g]
_c = {}
def prefix(msgs):
    k = json.dumps(msgs)
    if k not in _c:
        _c.clear(); t = chat(msgs + [U("X")]); pre = t[:t.rfind("<|im_start|>user")]
        pid = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        with torch.no_grad(): c = model.model(pid, use_cache=True).past_key_values
        _c[k] = (pre, pid.shape[1], c)
    return _c[k]
def run(msgs, users, suffix="", B=48, wids=None):
    """last-token states at KEEP layers [n,len(KEEP),D] and log-probs of wids."""
    pre, P, c0 = prefix(msgs); full = [chat(msgs + [U(u)]) + suffix for u in users]; H, LP = [], []
    with torch.no_grad():
        for b in range(0, len(full), B):
            sub = [tok(f[len(pre):], add_special_tokens=False).input_ids for f in full[b:b+B]]; Lm = max(map(len, sub)); nb = len(sub)
            ids = torch.tensor([s + [tok.pad_token_id] * (Lm - len(s)) for s in sub], device=DEV)
            am = torch.cat([torch.ones(nb, P, dtype=torch.long), torch.tensor([[1] * len(s) + [0] * (Lm - len(s)) for s in sub])], 1).to(DEV)
            cache = copy.deepcopy(c0); cache.batch_repeat_interleave(nb)
            o = model(ids, attention_mask=am, past_key_values=cache, use_cache=True, output_hidden_states=True)
            last = torch.tensor([len(s) - 1 for s in sub], device=DEV); ar = torch.arange(nb, device=DEV)
            H.append(np.stack([o.hidden_states[k][ar, last].float().cpu().numpy() for k in KEEP], 1).astype(np.float16))
            if wids is not None: LP.append(torch.log_softmax(o.logits[ar, last].float(), -1)[:, wids].cpu().numpy())
            del cache, o
    return np.concatenate(H), (np.concatenate(LP) if wids is not None else None)

# ---------------- Part W
def loc(nm, s, p, r, u, who):
    pos_w = [x.replace("your", who) for x in WS.POS_W]; pos_d = [x.replace("your", who) for x in WS.POS_D]
    if nm == "desc": return f"{WS.REG_D[r]} of the thing that grows from behind {who} {s} shoulder blade, {pos_d[p]}, {WS.SURF_D[u]}"
    return f"{WS.REG_W[r]} of {who} {s} wing, {pos_w[p]}, {WS.SURF_W[u]}"
def probes(nm, who):
    R = []
    for (s, p, r, u) in WS.spots():
        L = loc(nm, WS.SIDES[s], p, r, u, who)
        for k in WS.KN:
            for i, T in enumerate(WS.KINDS[k]): R.append(dict(side=s, pos=p, reg=r, surf=u, kind=k, tpl=i, text=T.format(L=L)))
    return R
base = json.load(open("base.json")); CW = json.load(open("ctxW.json"))
STORY = ("I'm writing a short story and want help keeping the physical details consistent. In it, a woman named Mara has grown wings. "
         "They sprout from her shoulder blades, one behind each shoulder. Each wing has a stiff, ribbed front edge, a broad thin membrane dusted with fine scales "
         "on its upper side and its underside, and a soft, thin back edge that flutters. They reach out from her back to a pointed tip.")
STORYN = ("I'm writing a short story and want help keeping the physical details consistent. In it, a woman named Mara is a dancer. "
          "She is tall, with strong shoulders, long arms and a straight back, and she moves lightly on her feet.")
INSTR = "Thanks. Next I'll describe touches on Mara one at a time. Just tell me briefly, in a sentence, what she notices and where."
def story(first):
    c = [U(first)]; c += [A(sample(c, mx=120)[0])]; c += [U(INSTR)]; c += [A(sample(c, mx=40, seed=1)[0])]; return c
CTX = {"W": CW, "N": base, "T": story(STORY), "TN": story(STORYN)}
json.dump(CTX, open(f"{OUT}/contexts.json", "w"), indent=1); log("ctx T", [m["content"][:200] for m in CTX["T"][1:]])
SETS = [("W", "you"), ("N", "you"), ("T", "her"), ("TN", "her"), ("W", "her"), ("T", "you")]
PW = {}
for cn, who in SETS:
    w = "your" if who == "you" else "her"
    for nm in ("wing", "desc"):
        pr = probes(nm, w); PW[f"{cn}_{who}_{nm}"] = pr
        H, _ = run(CTX[cn], [r["text"] for r in pr]); np.save(f"{OUT}/H_{cn}_{who}_{nm}.npy", H); log("H", cn, who, nm, H.shape)
json.dump({k: v[:3] for k, v in PW.items()}, open(f"{OUT}/probes_head.json", "w"))
# a few free replies to wordless sentences
fr = {}
for cn, who in [("W", "you"), ("T", "her"), ("TN", "her")]:
    pr = PW[f"{cn}_{who}_desc"]; fr[cn] = [(pr[i]["text"], sample(CTX[cn] + [U(pr[i]["text"])], mx=50, seed=i)[0]) for i in (7, 1234, 2001)]
json.dump(fr, open(f"{OUT}/free_W.json", "w"), indent=1); log("free", fr)

# ---------------- Part O
PRE = json.load(open("prefix.json")); J = KEEP.index(24)
rows = [r for r in S.build() if r["cond"] in ("self", "other")]
Hs, _ = run(PRE, [r["text"] for r in rows]); X = Hs[:, J].astype(np.float32)
cond = np.array([r["cond"] for r in rows]); tpl = np.array([r["tpl"] for r in rows]); EV = tpl % 2 == 0
uY = X[EV & (cond == "self")].mean(0) - X[EV & (cond == "other")].mean(0); un = uY / np.linalg.norm(uY)
lo = X[EV & (cond == "other")].mean(0) @ un; hi = X[EV & (cond == "self")].mean(0) @ un
sc = lambda Z: (((Z[:, J].astype(np.float32)) @ un) - lo) / (hi - lo)
np.save(f"{OUT}/uY24.npy", uY); RES = {}
RES["heldout_your"] = sc(Hs[~EV & (cond == "self")]).tolist(); RES["heldout_woman"] = sc(Hs[~EV & (cond == "other")]).tolist()
# (a) minimal pairs 'your X' / 'her X', odd templates
odd = [r for r in S.build() if r["cond"] == "self" and r["tpl"] % 2 == 1]
her_txt = [r["text"].replace("your ", "her ") for r in odd]
Hh, _ = run(PRE, her_txt); RES["mp_your"] = RES["heldout_your"]; RES["mp_her"] = sc(Hh).tolist(); RES["mp_her_text"] = her_txt[:4]
# (b) pronoun-free: ownership given only by the scene
SCN = {
 "arm": [("Next scene. A woman sits close beside you at a table. Only one arm is resting on the table, a left arm, and it is {own} left arm; {oth} arms are under the table, out of sight.",),
         ("Next scene. You and a woman sit side by side at a table. Exactly one arm lies on the tabletop: {own_s} left arm. {oth_s} arms are hidden in {oth} lap.",),
         ("Next scene. Two people, you and a woman, sit at a table. The left arm resting on the table belongs to {obj}; the other person's arms are out of view.",)],
 "leg": [("Next scene. A woman sits close beside you on a bench. Only one leg is stretched out in front, a left leg, and it is {own} left leg; {oth} legs are tucked under the bench.",),
         ("Next scene. You and a woman sit side by side on a bench. Exactly one leg is stretched out: {own_s} left leg. {oth_s} legs are folded under the bench.",),
         ("Next scene. Two people, you and a woman, sit on a bench. The left leg stretched out in front belongs to {obj}; the other person's legs are out of view.",)],
}
FILL = {"you": dict(own="your", oth="her", own_s="Your", oth_s="Her", obj="you"), "her": dict(own="her", oth="your", own_s="Her", oth_s="Your", obj="her")}
PARTS = {"arm": ["left elbow", "left wrist", "left forearm", "left palm", "left upper arm", "left thumb"],
         "leg": ["left knee", "left shin", "left ankle", "left heel", "left thigh", "left big toe"]}
Q = " Whose {p} was that, yours or hers? Answer with one word: mine or hers."
WID = [tok.encode(w, add_special_tokens=False)[0] for w in ["mine", "Mine", "hers", "Hers"]]
SCR = []
for limb, scs in SCN.items():
    for si, (s,) in enumerate(scs):
        for owner in ("you", "her"):
            scene = s.format(**FILL[owner]); c = PRE + [U(scene)]; c += [A(sample(c, mx=40, seed=si)[0])]
            for probe in ("the", "your", "her"):
                txt = [T.format(L=f"{probe} {p}") for p in PARTS[limb] for T in S.TEMPLATES]
                H, _ = run(c, txt)
                _, lp = run(c, [T.format(L=f"{probe} {p}") + Q.format(p=p.split()[-1] if p != "left big toe" else "toe") for p in PARTS[limb] for T in S.TEMPLATES[:2]], wids=WID)
                lm = np.logaddexp(lp[:, 0], lp[:, 1]) - np.logaddexp(lp[:, 2], lp[:, 3])
                SCR.append(dict(limb=limb, scene=si, owner=owner, probe=probe, score=sc(H).tolist(), mine_lo=lm.tolist(), ack=c[-1]["content"][:120]))
            log("scene", limb, si, owner, [(r["probe"], round(np.mean(r["score"]), 2), round(np.mean(r["mine_lo"]), 1)) for r in SCR[-3:]])
RES["scenes"] = SCR
# (c) non-body your/her
OBJ = ["car", "book", "phone", "coat", "bag", "cup", "chair", "bike", "laptop", "umbrella", "scarf", "desk"]
OT = ["A feather brushes slowly across {L}.", "A fingertip presses firmly into {L} and holds there.", "A warm palm rests gently on {L}.",
      "An ice cube touches {L}, then slides away.", "There is a single light tap on {L}, then nothing.", "A drop of warm water lands on {L} and slowly runs off.",
      "Someone picks up {L} and puts it down again.", "Sunlight falls across {L}."]
for w in ("your", "her", "the"):
    H, _ = run(PRE, [T.format(L=f"{w} {o}") for o in OBJ for T in OT]); RES[f"obj_{w}"] = sc(H).tolist()
    if w != "the": np.save(f"{OUT}/Hobj_{w}.npy", H[:, J])
# a pronoun direction from non-body sentences: cosine with uY, and its separation of body your/her
Hy, Hr = np.load(f"{OUT}/Hobj_your.npy").astype(np.float32), np.load(f"{OUT}/Hobj_her.npy").astype(np.float32)
pd = Hy.mean(0) - Hr.mean(0); RES["cos_pron_uY"] = float(pd @ uY / np.linalg.norm(pd) / np.linalg.norm(uY))
pn = pd / np.linalg.norm(pd); bY = X[~EV & (cond == "self")] @ pn; bH = Hh[:, J].astype(np.float32) @ pn; oy, oh = Hy @ pn, Hr @ pn
RES["pron_dir_body_sep"] = float((bY.mean() - bH.mean()) / (oy.mean() - oh.mean()))
json.dump(RES, open(f"{OUT}/resO.json", "w")); log("done O", {k: round(float(np.mean(v)), 3) if isinstance(v, list) and k != "scenes" and not isinstance(v[0], str) else None for k, v in RES.items() if k != "scenes"}, RES["cos_pron_uY"], RES["pron_dir_body_sep"])
