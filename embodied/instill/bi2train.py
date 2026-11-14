"""v2: linear adapter with norm cap; joints and touch at separate tokens; continuous + neighbourhood questions.
Trained only on uncrossed poses (crossed poses held out for the crossed-hands test)."""
import sys, time, json, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import body, body2

DEV = "mps"; MODEL = "Qwen/Qwen3-0.6B"; T = 8; L_IN = 6; L_FWD = 14
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; BS = 24
import os
NAMELESS = os.environ.get("NAMELESS") == "1"  # forward-model loss only, no words about the body
TAG = "3" if NAMELESS else "2"
PH_J, PH_T = 3968, 6306  # ' ~' joint token, ' ^' touch token
CAP = 1.0  # injected-vector norm cap, as a multiple of the median residual norm at the hook


class Adapter(nn.Module):
    """Two linear maps (joints -> joint token, touch -> touch token), each norm-capped."""
    def __init__(s, d=1024):
        super().__init__()
        s.j = nn.Linear(12, d); s.t = nn.Linear(body.N_SKIN, d)
        s.register_buffer("cap", torch.tensor(1.0))
    def _clip(s, v):
        n = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return v * torch.clamp(s.cap / n, max=1.0)
    def forward(s, x, kind):  # x [...,36]; kind [...] 1=joint 2=touch 0=none
        vj, vt = s._clip(s.j(x[..., :12])), s._clip(s.t(x[..., 12:]))
        return vj * (kind == 1)[..., None] + vt * (kind == 2)[..., None]


class Body:
    def __init__(s, lora=True):
        s.tok = AutoTokenizer.from_pretrained(MODEL)
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        for p in m.parameters(): p.requires_grad_(False)
        if lora:
            m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             layers_to_transform=list(range(L_IN, 28))))
        s.m = m.to(DEV); s.ad = Adapter().to(DEV)
        s.fwd = nn.Linear(1024, body.SENSOR_DIM).to(DEV)
        s.inject = None; s.extra = None; s.resid_norm = None
        s.layers = s.m.get_base_model().model.layers if lora else s.m.model.layers
        s.layers[L_IN - 1].register_forward_hook(s._hook)

    def _hook(s, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if s.inject is None:
            if s.resid_norm == "measure": s._rn = h[:, 1:].norm(dim=-1).median().item()
            return out
        kind, vec = s.inject
        h = h + vec * (kind > 0)[..., None]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def build(s, S_batch, qtexts, answers=None):
        ids, lab, kinds, tidx = [], [], [], []
        pre = s.tok.encode("Body:") + [PH_J, PH_T] * T
        for i, q in enumerate(qtexts):
            p = pre + s.tok.encode(f"\nQ: {q}\nA:")
            a = s.tok.encode(" " + answers[i]) if answers is not None else []
            ids.append(p + a); lab.append([-100] * len(p) + a)
            kinds.append([0] + [1, 2] * T + [0] * (len(p) - 1 - 2 * T + len(a)))
        n = max(map(len, ids)); pad = s.tok.pad_token_id
        lens = torch.tensor([len(x) for x in ids], device=DEV)
        f = lambda xs, v: torch.tensor([x + [v] * (n - len(x)) for x in xs], device=DEV)
        ids, lab, kind = f(ids, pad), f(lab, -100), f(kinds, 0)
        att = (torch.arange(n, device=DEV)[None] < lens[:, None]).long()
        Sx = torch.zeros(len(qtexts), n, body.SENSOR_DIM, device=DEV)
        Ss = torch.tensor(np.stack(S_batch), device=DEV)
        Sx[:, 1:1 + 2 * T:2] = Ss; Sx[:, 2:2 + 2 * T:2] = Ss
        return ids, lab, kind, att, Sx

    def run(s, ids, kind, att, Sx, ablate=False, hidden=False):
        vec = s.ad(Sx, kind)
        if ablate: vec = vec * 0
        if s.extra is not None: vec = vec + s.extra
        s.inject = (kind, vec)
        out = s.m(input_ids=ids, attention_mask=att, output_hidden_states=hidden)
        s.inject = None
        return out

    @torch.no_grad()
    def calibrate(s):
        """Cap = median residual norm at the hook on placeholder-token prompts."""
        rng = np.random.default_rng(99); S = [body2.episode(rng, T)[0] for _ in range(8)]
        ids, lab, kind, att, Sx = s.build(S, ["Where are you being touched?"] * 8)
        s.resid_norm = "measure"; s.m(input_ids=ids, attention_mask=att); s.resid_norm = None
        s.ad.cap.fill_(CAP * s._rn); return s._rn


# token positions: joint token of step t = 1+2t, touch token = 2+2t
JTOK = lambda t: 1 + 2 * t
TTOK = lambda t: 2 + 2 * t


def batch(rng, n):
    S, Q, A, C, I = [], [], [], [], []
    for _ in range(n):
        s, info = body2.episode(rng, T, want_crossed=False)
        qa = body2.questions(info); q, a, c = qa[rng.integers(len(qa))]
        S.append(s); Q.append(q); A.append(a); C.append(c); I.append(info)
    return S, Q, A, C, I


@torch.no_grad()
def cand_logp(B, S, Q, C):
    """Total log-prob of each candidate answer, per item."""
    out = []
    for s, q, cands in zip(S, Q, C):
        ids, lab, kind, att, Sx = B.build([s] * len(cands), [q] * len(cands), cands)
        lp = torch.log_softmax(B.run(ids, kind, att, Sx).logits[:, :-1].float(), -1)
        tgt = lab[:, 1:]; m = tgt != -100
        out.append((lp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0] * m).sum(1).cpu().numpy())
    return out


def score(B, S, Q, C):
    return [c[int(lp.argmax())] for c, lp in zip(C, cand_logp(B, S, Q, C))]


def evaluate(B, n=120, seed=123, crossed=False, ablate=False):
    rng = np.random.default_rng(seed); acc = {}; S, Q, A, C = [], [], [], []
    for _ in range(n):
        s, info = body2.episode(rng, T, want_crossed=crossed)
        for q, a, c in body2.questions(info):
            S.append(s); Q.append(q); A.append(a); C.append(c)
    if ablate:  # zero the adapter output
        sd = {k: v.clone() for k, v in B.ad.state_dict().items()}
        for p in B.ad.parameters(): p.data.zero_()
    P = score(B, S, Q, C)
    if ablate: B.ad.load_state_dict(sd)
    for q, a, p in zip(Q, A, P):
        k = q.replace("left ", "").replace("right ", "") if "near your" in q else q
        acc.setdefault(k, []).append(abs(int(a) - int(p)) if a.isdigit() else a == p)
    return {(q + (" [mean abs error, tenths]" if "(10)" in q else "")): round(float(np.mean(v)), 3) for q, v in acc.items()}


if __name__ == "__main__":
    torch.manual_seed(0); rng = np.random.default_rng(0)
    B = Body(); print("resid norm at hook", B.calibrate(), flush=True)
    params = [{"params": B.ad.parameters(), "lr": 1e-3}, {"params": B.fwd.parameters(), "lr": 1e-3},
              {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": 2e-4}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    log = []; t0 = time.time(); start = 0; RS = f"resume{TAG}.pt"
    if os.path.exists(RS):  # resume after an external kill
        r = torch.load(RS, weights_only=False); B.ad.load_state_dict(r["ad"]); B.fwd.load_state_dict(r["fwd"])
        B.m.load_state_dict(r["lora"], strict=False); opt.load_state_dict(r["opt"]); start, log = r["step"], r["log"]
        rng = np.random.default_rng(1000 + start); print("resumed at", start, flush=True)
    for step in range(start, STEPS):
        if step % 200 == 0 and step > start:
            torch.save({"ad": B.ad.state_dict(), "fwd": B.fwd.state_dict(), "opt": opt.state_dict(), "step": step, "log": log,
                        "lora": {k: v for k, v in B.m.state_dict().items() if "lora" in k}}, RS)
        S, Q, A, C, I = batch(rng, BS)
        ids, lab, kind, att, Sx = B.build(S, Q, A)
        out = B.run(ids, kind, att, Sx, hidden=True)
        lm = nn.functional.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
        h = out.hidden_states[L_FWD + 1][:, [TTOK(t) for t in range(T - 1)]]  # touch token of step t predicts sensors at t+1
        fl = ((B.fwd(h) - torch.tensor(np.stack(S), device=DEV)[:, 1:]) ** 2).mean()
        loss = fl if NAMELESS else lm + fl
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            log.append(dict(step=step, lm=lm.item(), fwd=fl.item(), t=time.time() - t0)); print(log[-1], flush=True)
    torch.save({"ad": B.ad.state_dict(), "fwd": B.fwd.state_dict(),
                "lora": {k: v for k, v in B.m.state_dict().items() if "lora" in k}}, f"ckpt{TAG}.pt")
    if NAMELESS: json.dump(dict(log=log), open(f"train{TAG}_log.json", "w")); sys.exit()
    ev = evaluate(B); ev_x = evaluate(B, crossed=True); ev_abl = evaluate(B, ablate=True)
    print("eval", ev, "\ncrossed", ev_x, "\nablated", ev_abl, flush=True)
    json.dump(dict(log=log, eval=ev, eval_crossed=ev_x, ablated=ev_abl, resid_norm=float(B.ad.cap)), open(f"train{TAG}_log.json", "w"), indent=1)
