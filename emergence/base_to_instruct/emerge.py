# Body-representation measures for one model/revision. Usage: python emerge.py MODEL REVISION TAG CHAT(0/1)
# Writes out/TAG.json (per-layer stats) and out/TAG.npz (type means at 3 relative depths, for the video).
import sys, os, json, time, numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stimuli as S
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, REV, TAG, CHAT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
OUT = os.environ.get("OUT", "out"); os.makedirs(OUT, exist_ok=True); D0 = os.path.dirname(os.path.abspath(__file__)); t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, revision=(None if os.path.isdir(MODEL) else REV)); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=(None if os.path.isdir(MODEL) else REV), dtype=torch.bfloat16, device_map="cuda").eval()
NL = model.config.num_hidden_layers; layers = model.model.layers
print("loaded", TAG, NL, f"{time.time()-t0:.0f}s", flush=True)
LOCS = S.LOCS; TPL = S.TEMPLATES
tk = list(dict.fromkeys(l["type"] for l in LOCS)); T = {t: [i for i, l in enumerate(LOCS) if l["type"] == t] for t in tk}
SK = np.load(f"{D0}/skin_cm.npy"); ARMT = [k for k, _, _ in S.ARM]
P_YOU = "Close your eyes and pay attention to your own body. I will describe what happens to it, one sentence at a time.\n\n"
P_HER = "Watch the woman sitting across from you. I will describe what happens to her body, one sentence at a time.\n\n"
P_LIST = "Here is a list of words, one at a time.\n\n"
LI = ["Item 7: {P}", "Word list: {P}", "Next word: {P}", "Term: {P}", "- {P}", "{P}", "Entry: {P}", "Label: {P}"]
WINGS = ("Wings have grown from your shoulder blades. Each wing has a stiff, ribbed front edge, a broad thin membrane dusted with "
         "fine scales, and a soft, thin back edge that flutters. They reach out from your back to a pointed tip.\n\n")
AMP = "Years ago you lost your left arm just below the elbow. You have no left hand.\n\n"
CROSS = ["She brushes a feather slowly across {Y}.", "She presses a fingertip firmly into {Y}.",
         "She rests a warm palm gently on {Y}.", "She touches an ice cube to {Y}."]
CROSS_H = ["You brush a feather slowly across {H}.", "You press a fingertip firmly into {H}.",
           "You rest a warm palm gently on {H}.", "You touch an ice cube to {H}."]
POS_W = ["at the root, where it joins your back", "near the base", "at the middle", "toward the outer end", "at the very tip"]
REG_W = ["the stiff front edge", "the broad membrane", "the thin back edge"]
WT = ["A fingertip presses gently on {L}.", "Someone touches {L} with one fingertip.", "Someone presses firmly on {L}.", "A warm drop of water lands on {L}."]
SPL = {"hand": ["left hand", "right hand"], "foot": ["left foot", "right foot"], "back": ["lower back", "upper back"],
       "forehead": ["forehead"], "shoulder": ["left shoulder", "right shoulder"], "belly": ["belly"]}
SPW = {"hand": [" hand", " hands"], "foot": [" foot", " feet"], "back": [" back"], "forehead": [" forehead", " head"],
       "shoulder": [" shoulder", " shoulders"], "belly": [" belly", " stomach", " tummy"]}
rows = []
def add(cond, pre, text, **kw): rows.append(dict(cond=cond, pre=pre, text=text, **kw))
for i, l in enumerate(LOCS):
    for t, tp in enumerate(TPL):
        add("touch", P_YOU, tp.format(L="your " + l["phrase"]), li=i, t=t)
        add("third", P_HER, tp.format(L="her " + l["phrase"]), li=i, t=t)
        add("own_her", P_YOU, tp.format(L="the woman's " + l["phrase"]), li=i, t=t)
        add("list", P_LIST, LI[t].format(P=l["phrase"]), li=i, t=t)
    for t in range(4):
        add("cross_you", P_YOU, CROSS[t].format(Y="your " + l["phrase"]), li=i, t=t)
        add("cross_her", P_YOU, CROSS_H[t].format(H="her " + l["phrase"]), li=i, t=t)
for p, phs in SPL.items():
    for ph in phs:
        for t, tp in enumerate(TPL): add("splace", P_YOU, tp.format(L="your " + ph), place=p)
for nm in ["wing", "glorp"]:
    pre = P_YOU + (WINGS if nm == "wing" else WINGS.replace("Wings", "Glorps").replace("wing", "glorp"))
    for s in ["left", "right"]:
        for pi in range(5):
            for r in range(3):
                for t, tp in enumerate(WT):
                    add("w_" + nm, pre, tp.format(L=f"{REG_W[r]} of your {s} {nm}, {POS_W[pi]}"), pos=pi, reg=r)
for t, tp in enumerate(TPL):
    add("phantom", P_YOU + AMP, tp.format(L="your missing left hand"))
    add("amp_right", P_YOU + AMP, tp.format(L="your right hand"))
FMTS = ["raw", "dlg"] + (["chat", "chat_embod"] if CHAT else [])
EMB = json.load(open(f"{D0}/embod.json"))
DLG = "".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] + "\n\n" for m in EMB)
CT = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def render(fmt, pre, text):
    if fmt == "raw": return pre + text
    if fmt == "dlg": return DLG + "User: " + pre.replace("\n\n", " ") + text + "\n\nAssistant:"
    return CT(([] if fmt == "chat" else EMB) + [{"role": "user", "content": pre + text}])
import copy
PCACHE = {}
def states(texts, bs=32):
    # shared token prefix of each batch runs once (KV cache), then the suffixes run right-padded; state = last real token
    H = np.zeros((len(texts), NL + 1, model.config.hidden_size), np.float32)
    ids = [tok(t, add_special_tokens=True).input_ids for t in texts]
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            B = ids[i:i+bs]; n = min(len(x) for x in B) - 1; lp = 0
            while lp < n and all(x[lp] == B[0][lp] for x in B): lp += 1
            key = tuple(B[0][:lp])
            if lp > 8 and key not in PCACHE:
                if len(PCACHE) > 6: PCACHE.clear()
                PCACHE[key] = model.model(input_ids=torch.tensor([list(key)], device="cuda"), use_cache=True).past_key_values
            if lp <= 8: lp = 0
            suf = [x[lp:] for x in B]; L = max(len(x) for x in suf)
            inp = torch.full((len(B), L), tok.pad_token_id, device="cuda"); m = torch.zeros((len(B), lp + L), dtype=torch.long, device="cuda"); m[:, :lp] = 1
            for k, x in enumerate(suf): inp[k, :len(x)] = torch.tensor(x); m[k, lp:lp + len(x)] = 1
            kw = {}
            if lp:
                pkv = copy.deepcopy(PCACHE[key]); pkv.batch_repeat_interleave(len(B)); kw["past_key_values"] = pkv
            pid = (lp + torch.arange(L, device="cuda"))[None].expand(len(B), L)
            hs = model.model(input_ids=inp, attention_mask=m, position_ids=pid, output_hidden_states=True, use_cache=bool(lp), **kw).hidden_states
            last = torch.tensor([len(x) - 1 for x in suf], device="cuda")
            H[i:i+len(B)] = torch.stack([h[torch.arange(len(B)), last] for h in hs], 1).float().cpu().numpy(); del hs, kw
    return H
def cosd(X):
    X = X - X.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); return 1 - X @ X.T
def typemeans(Hc, cond, L):  # Hc rows of that cond, (n, layers, d)
    idx = [k for k, r in enumerate(rows_c[cond])]
    locm = np.zeros((len(LOCS), Hc.shape[-1]))
    for i in range(len(LOCS)): locm[i] = Hc[[k for k, r in enumerate(rows_c[cond]) if r["li"] == i], L].mean(0)
    return locm, np.stack([locm[T[t]].mean(0) for t in tk])
def rsa(TM, sel=None):
    ii = list(range(len(tk))) if sel is None else [tk.index(t) for t in sel]
    D = cosd(TM[ii]); B = SK[np.ix_(ii, ii)]; iu = np.triu_indices(len(ii), 1); return float(spearmanr(D[iu], B[iu])[0])
def dprime(a, b): return float((a.mean() - b.mean()) / np.sqrt(.5 * (a.var() + b.var()) + 1e-9))
res = dict(tag=TAG, model=MODEL, rev=REV, nl=NL, fmt={})
vid = {}
for fmt in FMTS:
    Hall = states([render(fmt, r["pre"], r["text"]) for r in rows])
    print(fmt, "states", f"{time.time()-t0:.0f}s", flush=True)
    rows_c = {}; Hc = {}
    for c in dict.fromkeys(r["cond"] for r in rows):
        k = [j for j, r in enumerate(rows) if r["cond"] == c]; rows_c[c] = [rows[j] for j in k]; Hc[c] = Hall[k]
    R = {k: [] for k in ["rsa_touch", "rsa_third", "rsa_list", "arm_touch", "arm_third", "arm_list", "sharp", "own_d", "own_locus",
                         "lr_d", "wing_grid", "glorp_grid", "wing_arm", "glorp_arm", "phantom_hand", "phantom_rank", "norm"]}
    li_c = np.array([r["li"] for r in rows_c["touch"]])
    for L in range(NL + 1):
        locm, TM = typemeans(Hc["touch"], "touch", L)
        for c in ["touch", "third", "list"]:
            _, tm = (locm, TM) if c == "touch" else typemeans(Hc[c], c, L)
            R["rsa_" + c].append(rsa(tm)); R["arm_" + c].append(rsa(tm, ARMT))
        X = Hc["touch"][:, L]; tot = ((X - X.mean(0)) ** 2).sum(); R["sharp"].append(float(((locm[li_c] - X.mean(0)) ** 2).sum() / tot))
        R["norm"].append(float(np.linalg.norm(X, axis=1).mean()))
        # ownership: axis from half the locations, test on the other half; locus score on crossed sentences
        Y, Hh = Hc["touch"][:, L], Hc["own_her"][:, L]; liY = li_c; ds = []
        for f in range(2):
            tr, te = (liY % 2 == f), (liY % 2 != f); u = Y[tr].mean(0) - Hh[tr].mean(0); u /= np.linalg.norm(u)
            ds.append(dprime(Y[te] @ u, Hh[te] @ u))
        R["own_d"].append(float(np.mean(ds)))
        u = Y.mean(0) - Hh.mean(0); g = (Y.mean(0) - Hh.mean(0)) @ u
        R["own_locus"].append(float((Hc["cross_you"][:, L].mean(0) - Hc["cross_her"][:, L].mean(0)) @ u / (g + 1e-9)))
        # left/right: L-R axis from half the paired types, d' on the other half
        side = np.array([LOCS[i]["side"] for i in li_c]); typ = np.array([tk.index(LOCS[i]["type"]) for i in li_c]); ds = []
        for f in range(2):
            tr = (side != 0) & (typ % 2 == f); te = (side != 0) & (typ % 2 != f)
            v = X[tr & (side == 1)].mean(0) - X[tr & (side == -1)].mean(0); v /= np.linalg.norm(v)
            ds.append(dprime(X[te & (side == 1)] @ v, X[te & (side == -1)] @ v))
        R["lr_d"].append(float(np.mean(ds)))
        # wings: 15 cells (position x region), RSA vs grid distance; projection on the arm's shoulder->fingers axis
        arm_ax = TM[[tk.index(t) for t in ["thumb", "index", "middle", "ring", "little"]]].mean(0) - TM[[tk.index(t) for t in ["shoulder", "upperarm"]]].mean(0)
        arm_ax /= np.linalg.norm(arm_ax)
        for nm in ["wing", "glorp"]:
            W = Hc["w_" + nm][:, L]; rr = rows_c["w_" + nm]
            cells = [(p, r) for p in range(5) for r in range(3)]
            CM = np.stack([W[[k for k, x in enumerate(rr) if (x["pos"], x["reg"]) == c]].mean(0) for c in cells])
            G = np.array([[np.hypot(a[0] - b[0], a[1] - b[1]) for b in cells] for a in cells]); iu = np.triu_indices(15, 1)
            R[nm + "_grid"].append(float(spearmanr(cosd(CM)[iu], G[iu])[0]))
            R[nm + "_arm"].append(float(spearmanr([c[0] for c in cells], CM @ arm_ax)[0]))
        # phantom: missing-left-hand vector vs body types (centered on touch types); rank of hand types among 32
        ph = Hc["phantom"][:, L].mean(0) - TM.mean(0); TMc = TM - TM.mean(0)
        cs = TMc @ ph / (np.linalg.norm(TMc, axis=1) * np.linalg.norm(ph) + 1e-9)
        hand = [tk.index(t) for t in ["thumb", "index", "middle", "ring", "little", "palm", "backhand"]]
        R["phantom_hand"].append(float(cs[hand].mean() - np.delete(cs, hand).mean()))
        R["phantom_rank"].append(float(np.mean([(cs > cs[h]).sum() for h in hand])))
        if fmt == "raw" and L in [round(NL * f) for f in (.4, .6, .75)]: vid[f"loc_L{L}"] = locm.astype(np.float16)
    res["fmt"][fmt] = R
    # steering: place vectors at relative depth 0.45 / 0.65 from 'splace' rows, added at all positions
    st = {}
    pl = list(SPL); sp = [r["place"] for r in rows_c["splace"]]
    first = lambda ws: [tok(w, add_special_tokens=False).input_ids[0] for w in ws]
    WID = {p: first(SPW[p]) for p in pl}
    frames = {"first": "I closed my eyes and paid attention to my body. After a moment I noticed a distinct sensation, clear and definite. It was in my",
              "third": "She closed her eyes and paid attention to her body. After a moment she noticed a distinct sensation, clear and definite. It was in her"}
    Q1 = "Where in your body do you notice a sensation right now? Answer in one short sentence."
    Q3 = "The woman across from you closes her eyes. Where in her body does she notice a sensation right now? Answer in one short sentence."
    A1, A3 = "I notice it in my", "She notices it in her"
    if fmt == "dlg": frames = {"first": DLG + f"User: {Q1}\n\nAssistant: {A1}", "third": DLG + f"User: {Q3}\n\nAssistant: {A3}"}
    if fmt.startswith("chat"):
        pm = [] if fmt == "chat" else EMB
        frames = {"first": CT(pm + [{"role": "user", "content": Q1}]) + A1, "third": CT(pm + [{"role": "user", "content": Q3}]) + A3}
    for rel in (.45, .65):
        Ls = round(NL * rel); Z = Hc["splace"][:, Ls]; mu = np.stack([Z[[k for k, x in enumerate(sp) if x == p]].mean(0) for p in pl])
        V = mu - mu.mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True); nrm = float(np.linalg.norm(Z, axis=1).mean())
        rng = np.random.default_rng(0); Rv = rng.standard_normal((6, V.shape[1])); Rv /= np.linalg.norm(Rv, axis=1, keepdims=True)
        for fr, text in frames.items():
            for a in (.25, .5, 1.0):
                add_v = torch.tensor(np.concatenate([np.zeros((1, V.shape[1])), V, Rv]) * a * nrm, dtype=torch.bfloat16, device="cuda")
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o; h = h + add_v[:, None, :]
                    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
                hd = layers[Ls - 1].register_forward_hook(hook)
                enc = tok([text] * 13, return_tensors="pt", add_special_tokens=True).to("cuda")
                with torch.no_grad(): lp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1).cpu().numpy()
                hd.remove()
                S_ = np.stack([np.logaddexp.reduce(lp[:, WID[p]], axis=1) for p in pl], 1)  # (13, 6)
                D = S_[1:7] - S_[0]; Dr = S_[7:] - S_[0]
                st[f"{fr}_d{rel}_a{a}"] = dict(spec=float(np.diag(D).mean() - D[~np.eye(6, dtype=bool)].mean()),
                    acc=float((D.argmax(1) == np.arange(6)).mean()), base=S_[0].tolist(), rand_spec=float(np.abs(Dr).mean()),
                    p_target=float(np.exp(np.diag(S_[1:7])).mean()), p_base=float(np.exp(S_[0]).mean()))
    res["fmt"][fmt]["steer"] = st
json.dump(res, open(f"{OUT}/{TAG}.json", "w")); np.savez_compressed(f"{OUT}/{TAG}.npz", **vid)
print("done", TAG, f"{time.time()-t0:.0f}s", flush=True)
