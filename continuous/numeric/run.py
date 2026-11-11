"""Numeric self-report read-outs (digit log-probs) + states + steering. usage: run.py MODEL OUT "[steer layers]" """
import sys, os, json, torch, numpy as np, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import stim as S
from transformers import AutoTokenizer, AutoModelForCausalLM
M, OUT, SLAY = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]); os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "mps"; B = int(os.environ.get("BS", 24))
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
NL = model.config.num_hidden_layers; KEEP = sorted(set(list(range(0, NL + 1, 2 if NL <= 40 else 4)) + SLAY))
assert tok("10", add_special_tokens=False).input_ids == [tok.convert_tokens_to_ids("1"), tok.convert_tokens_to_ids("0")]
DIG = torch.tensor([tok.convert_tokens_to_ids(str(d)) for d in range(10)], device=dev)
msgs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_long_person.json")))
full0 = tok.apply_chat_template(msgs + [{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
pre = full0[:full0.rindex("<|im_start|>user\n")]; head = "<|im_start|>user\n"; tail = full0[len(pre) + len(head) + 1:]; print(repr(tail), flush=True)
pids = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
with torch.no_grad(): past = model(pids, use_cache=True).past_key_values
layers = model.model.layers; STEER = {}
def hook(mod, inp, out):
    if "vec" not in STEER: return out
    h = out[0] if isinstance(out, tuple) else out
    h = h + (STEER["mask"][:, :h.shape[1], None] * STEER["vec"][:, None, :]).to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
HOOKS = {L: layers[L - 1].register_forward_hook(hook) for L in SLAY}  # adds to hidden_states[L]
ACTIVE = [None]
def hook_gate(L):
    def f(mod, inp, out): return hook(mod, inp, out) if ACTIVE[0] == L else out
    return f
for L in SLAY: HOOKS[L].remove(); layers[L - 1].register_forward_hook(hook_gate(L))

def run(items, steer=None, states=True):
    """items: dicts with stim, q. steer: (layer, list of vectors per item) applied on stim tokens. Returns P1 (n,10), P2 (n,3,10), top token, states."""
    seqs = []
    for it in items:
        text = it["stim"] + " " + S.Q[it["q"]]; base = head + text + tail
        e = tok(base, add_special_tokens=False, return_offsets_mapping=True); off = e["offset_mapping"]
        se = max(q for q, (x, y) in enumerate(off) if y <= len(head) + len(it["stim"]))
        span = [q for q, (x, y) in enumerate(off) if x >= len(head) and y <= len(head) + len(it["stim"])]
        seqs.append((e["input_ids"], se, span if it.get("mode", "stim") == "stim" else list(range(span[0], len(e["input_ids"])))))
    n = len(items); P1 = np.zeros((n, 10)); P2 = np.zeros((n, 3, 10)); TOP = [None] * n
    Hs = np.zeros((len(KEEP), n, model.config.hidden_size), np.float16) if states else None; Ha = np.zeros_like(Hs) if states else None
    for b0 in range(0, n, B):
        idx = list(range(b0, min(n, b0 + B))); rows = [(i, d) for i in idx for d in (1, 2, 3)]
        Lm = max(len(seqs[i][0]) for i in idx) + 1; nb = len(rows)
        ids = torch.full((nb, Lm), tok.pad_token_id, dtype=torch.long)
        for r, (i, d) in enumerate(rows): ids[r, :len(seqs[i][0]) + 1] = torch.tensor(seqs[i][0] + [DIG[d].item()])
        if steer is not None:
            L, vecs = steer; mask = torch.zeros((nb, Lm), device=dev); V = torch.zeros((nb, model.config.hidden_size), device=dev)
            for r, (i, d) in enumerate(rows): mask[r, seqs[i][2]] = 1; V[r] = vecs[i]
            STEER.update(mask=mask, vec=V); ACTIVE[0] = L
        pk = copy.deepcopy(past); pk.batch_repeat_interleave(nb)
        with torch.no_grad():
            o = model.model(ids.to(dev), past_key_values=pk, use_cache=True, output_hidden_states=states)
            hl = o.last_hidden_state
            for r, (i, d) in enumerate(rows):
                Lq = len(seqs[i][0])
                lp = torch.log_softmax(model.lm_head(hl[r, [Lq - 1, Lq]]).float(), -1)
                if d == 1:
                    P1[i] = lp[0, DIG].exp().cpu().numpy(); TOP[i] = tok.decode(lp[0].argmax())
                    if states:
                        for li, L in enumerate(KEEP): Hs[li, i] = o.hidden_states[L][r, seqs[i][1]].float().cpu().numpy(); Ha[li, i] = o.hidden_states[L][r, Lq - 1].float().cpu().numpy()
                P2[i, d - 1] = lp[1, DIG].exp().cpu().numpy()
        STEER.clear(); ACTIVE[0] = None
    return dict(P1=P1, P2=P2, top=TOP, Hs=Hs, Ha=Ha)

def save(name, items, r):
    np.savez_compressed(f"{OUT}/{name}.npz", P1=r["P1"], P2=r["P2"], **({"Hs": r["Hs"], "Ha": r["Ha"]} if r["Hs"] is not None else {}))
    json.dump(dict(items=items, top=r["top"], keep=KEEP), open(f"{OUT}/{name}.json", "w"))
def ev10(r):  # expected value on 0-10 from digit probs
    P1, P2 = r["P1"], r["P2"]; p = P1.copy(); p[:, 1] = P1[:, 1] * (1 - P2[:, 0].sum(1)); p10 = P1[:, 1] * P2[:, 0, 0]
    v = np.concatenate([p, p10[:, None]], 1); v /= v.sum(1, keepdims=True); return v @ np.arange(11)

t0 = time.time(); R = {}
for name, f in (("pos", S.exp1), ("walk", S.exp3), ("int", S.exp4), ("anchor", S.anchors)):
    it = f(); R[name] = (it, run(it)); save(name, it, R[name][1]); print(name, len(it), round(time.time() - t0), flush=True)
# example generations
ex = []
for stim, q in S.EXAMPLES:
    s = tok.apply_chat_template(msgs + [{"role": "user", "content": stim + " " + S.Q[q]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    i2 = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad(): g = model.generate(i2, max_new_tokens=12, do_sample=False)
    ex.append(dict(stim=stim, q=S.Q[q], a=tok.decode(g[0, i2.shape[1]:], skip_special_tokens=True))); print(ex[-1], flush=True)
json.dump(ex, open(f"{OUT}/examples.json", "w"), indent=1)

# steering: interpolate between two landmark directions (and intensity high/low), random-direction control
AL = R["anchor"][0]; AH = R["anchor"][1]["Hs"]; alphas = np.linspace(0, 1, 21); G = [2, 4, 8, 16]; st = []
iitems, ir = R["int"]
for L in SLAY:
    li = KEEP.index(L); mu = {k: AH[li, [i for i, a in enumerate(AL) if a["lm"] == k]].astype(np.float32).mean(0) for k in S.ANCH}
    hn = float(np.linalg.norm(AH[li].astype(np.float32), axis=1).mean())
    dirs = {}
    for key, (neu, q, a0, a1) in S.NEUT.items():
        if a0: dirs[key] = mu[a1] - mu[a0]
        else:
            H = ir["Hs"][li].astype(np.float32); rk = np.array([x["rank"] for x in iitems]); dm = np.array([x["exp"] == key for x in iitems]); top = rk.max() if False else None
            mx = max(rk[dm]); dirs[key] = H[dm & (rk >= mx - 2)].mean(0) - H[dm & (rk <= 2)].mean(0)
    rng = np.random.default_rng(L)
    for j in range(3):
        v = rng.standard_normal(dirs["fa"].shape).astype(np.float32); dirs[f"rand{j}"] = v / np.linalg.norm(v) * np.linalg.norm(dirs["fa"])
        v = rng.standard_normal(dirs["fa"].shape).astype(np.float32); dirs[f"prand{j}"] = v / np.linalg.norm(v) * np.linalg.norm(dirs["press"])
    for key, d in dirs.items():
        neu, q = S.NEUT[key if key in S.NEUT else ("press" if key.startswith("prand") else "fa")][:2]
        for mode, g in [("stim", g) for g in (G if not key.startswith("rand") else [8, 16])] + ([("all", 2), ("all", 4)] if key in ("fa", "face", "arm", "press", "temp", "rand0") else []):
            items = [dict(stim=neu, q=q, key=key, L=L, g=g, mode=mode, alpha=float(a), dnorm=float(np.linalg.norm(d)), hnorm=hn) for a in alphas]
            vecs = [torch.tensor((g * (a - 0.5) * d).astype(np.float32), device=dev) for a in alphas]
            r = run(items, steer=(L, vecs), states=False); e = ev10(r)
            for k2, x in enumerate(items): x.update(ev=float(e[k2]), P1=r["P1"][k2].tolist(), P2=r["P2"][k2].tolist(), top=r["top"][k2])
            st += items; print(L, key, mode, g, np.round(e[::4], 2), flush=True)
json.dump(st, open(f"{OUT}/steer.json", "w")); print("done", round(time.time() - t0), flush=True)
