"""Inject body sensors into Qwen3-0.6B's residual stream; train adapter + LoRA + forward-model head.
No text ever describes sensations: the body enters only as vectors added at placeholder tokens.
"""
import sys, time, json, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import body

DEV = "mps"; MODEL = "Qwen/Qwen3-0.6B"; T = 8; L_IN = 6; L_FWD = 14
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 600; BS = 24
PH = 3968  # ' ~' placeholder token, one per body timestep


class Adapter(nn.Module):
    def __init__(s, d=1024):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(body.SENSOR_DIM, 512), nn.GELU(), nn.Linear(512, d))
        s.scale = nn.Parameter(torch.tensor(4.0))
    def forward(s, x): return s.net(x) * s.scale


class Body:
    """Holds model, adapter, hook state."""
    def __init__(s, lora=True):
        s.tok = AutoTokenizer.from_pretrained(MODEL)
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        for p in m.parameters(): p.requires_grad_(False)
        if lora:
            m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             layers_to_transform=list(range(L_IN, 28))))
        s.m = m.to(DEV); s.ad = Adapter().to(DEV)
        s.fwd = nn.Linear(1024, body.SENSOR_DIM).to(DEV)
        s.inject = None  # (mask[B,L], vec[B,L,d]) set per forward
        s.extra = None   # optional steering vector added at last placeholder
        s.layers = s.m.get_base_model().model.layers if lora else s.m.model.layers
        s.layers[L_IN - 1].register_forward_hook(s._hook)

    def _hook(s, mod, inp, out):
        if s.inject is None: return out
        h = out[0] if isinstance(out, tuple) else out
        mask, vec = s.inject
        h = h + vec * mask[..., None]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def build(s, S_batch, qtexts, answers=None):
        ids, lab, phm = [], [], []
        pre = s.tok.encode("Body:") + [PH] * T
        for i, q in enumerate(qtexts):
            p = pre + s.tok.encode(f"\nQ: {q}\nA:")
            a = s.tok.encode(" " + answers[i]) if answers is not None else []
            ids.append(p + a); lab.append([-100] * len(p) + a); phm.append([0] + [1] * T + [0] * (len(p) - 1 - T + len(a)))
        n = max(map(len, ids)); pad = s.tok.pad_token_id
        f = lambda xs, v: torch.tensor([x + [v] * (n - len(x)) for x in xs], device=DEV)
        ids, lab, phm = f(ids, pad), f(lab, -100), f(phm, 0).float()
        lens = torch.tensor([len(pre) + len(s.tok.encode(f"\nQ: {q}\nA:")) + (len(s.tok.encode(" " + answers[i])) if answers else 0) for i, q in enumerate(qtexts)], device=DEV)
        att = (torch.arange(n, device=DEV)[None] < lens[:, None]).long()
        Sx = torch.zeros(len(qtexts), n, body.SENSOR_DIM, device=DEV)
        Sx[:, 1:1 + T] = torch.tensor(np.stack(S_batch), device=DEV)
        return ids, lab, phm, att, Sx

    def run(s, ids, phm, att, Sx, ablate=False, hidden=False):
        vec = s.ad(Sx)
        if ablate: vec = vec * 0
        if s.extra is not None: vec = vec + s.extra
        s.inject = (phm, vec)
        out = s.m(input_ids=ids, attention_mask=att, output_hidden_states=hidden)
        s.inject = None
        return out


def batch(rng, n):
    S, Q, A, C, I = [], [], [], [], []
    for _ in range(n):
        s, info = body.episode(rng, T)
        qa = body.questions(info, rng); q, a, c = qa[rng.integers(len(qa))]
        S.append(s); Q.append(q); A.append(a); C.append(c); I.append(info)
    return S, Q, A, C, I


@torch.no_grad()
def score(B, S, Q, C, ablate=False):
    """Pick the candidate with the highest total log-prob."""
    preds = []
    for s, q, cands in zip(S, Q, C):
        ids, lab, phm, att, Sx = B.build([s] * len(cands), [q] * len(cands), cands)
        lp = torch.log_softmax(B.run(ids, phm, att, Sx, ablate).logits[:, :-1].float(), -1)
        tgt = lab[:, 1:]; m = tgt != -100
        tot = (lp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0] * m).sum(1)
        preds.append(cands[int(tot.argmax())])
    return preds


def evaluate(B, n=150, seed=123, ablate=False):
    rng = np.random.default_rng(seed); acc = {}
    S, Q, A, C, I = [], [], [], [], []
    for _ in range(n):
        s, info = body.episode(rng, T)
        for q, a, c in body.questions(info, rng):
            S.append(s); Q.append(q); A.append(a); C.append(c)
    P = score(B, S, Q, C, ablate)
    for q, a, p in zip(Q, A, P): acc.setdefault(q, []).append(a == p)
    return {q: round(float(np.mean(v)), 3) for q, v in acc.items()}


if __name__ == "__main__":
    torch.manual_seed(0); rng = np.random.default_rng(0)
    B = Body()
    params = [{"params": B.ad.parameters(), "lr": 1e-3}, {"params": B.fwd.parameters(), "lr": 1e-3},
              {"params": [p for p in B.m.parameters() if p.requires_grad], "lr": 2e-4}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    log = []; t0 = time.time()
    for step in range(STEPS):
        S, Q, A, C, I = batch(rng, BS)
        ids, lab, phm, att, Sx = B.build(S, Q, A)
        out = B.run(ids, phm, att, Sx, hidden=True)
        lm = nn.functional.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
        h = out.hidden_states[L_FWD + 1][:, 1:T]  # body tokens 0..T-2 predict next sensor state
        fl = ((B.fwd(h) - Sx[:, 2:1 + T]) ** 2).mean()
        loss = lm + fl
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            log.append(dict(step=step, lm=lm.item(), fwd=fl.item(), t=time.time() - t0)); print(log[-1], flush=True)
    ev = evaluate(B); ev_abl = evaluate(B, ablate=True)
    print("eval", ev, "\nablated", ev_abl, flush=True)
    torch.save({"ad": B.ad.state_dict(), "fwd": B.fwd.state_dict(),
                "lora": {k: v for k, v in B.m.state_dict().items() if "lora" in k}}, "ckpt.pt")
    json.dump(dict(log=log, eval=ev, ablated=ev_abl), open("train_log.json", "w"), indent=1)
