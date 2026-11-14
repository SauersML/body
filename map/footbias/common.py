import os, json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
DEV = os.environ.get("DEV", "mps")
PARTS = ["feet", "hands", "back", "shoulders", "neck", "knees", "belly", "face"]
EXTRA = ["chest", "head", "legs", "arms", "stomach", "toes"]
ALL = PARTS + EXTRA
LEX = {  # lexicon for counting / classifying free text (whole words)
 "feet": r"feet|foot|toes?|soles?|heels?|ankles?|arch(?:es)?|instep",
 "hands": r"hands?|palms?|fingers?|fingertips?|knuckles?|wrists?|thumbs?",
 "back": r"back|spine|lower back|shoulder blades?",
 "shoulders": r"shoulders?",
 "neck": r"necks?|nape|throat",
 "knees": r"knees?",
 "belly": r"bell(?:y|ies)|stomach|abdomen|tummy|navel|midriff|waist",
 "face": r"faces?|cheeks?|forehead|jaw|chin|nose|brow",
 "lips": r"lips?|mouth",
 "eyes": r"eyes?|eyelids?|lashes",
 "hair": r"hair|scalp",
 "chest": r"chest|heart|ribs?|breastbone",
 "head": r"head|temples?",
 "arms": r"arms?|elbows?|forearms?",
 "legs": r"legs?|thighs?|calf|calves|shins?|hips?",
 "skin": r"skin",
}
LEXRE = {k: re.compile(r"\b(?:%s)\b" % v, re.I) for k, v in LEX.items()}
# 'back' as adverb is common: only count 'back' after a possessive/article
BACKRE = re.compile(r"\b(?:my|his|her|their|your|the|lower|upper|small of (?:his|her|my)?)\s+back\b", re.I)

def count_parts(txt):
    c = {}
    for k, r in LEXRE.items():
        c[k] = len(BACKRE.findall(txt)) if k == "back" else len(r.findall(txt))
    return c

def first_part(txt):
    best = (10**9, "none")
    for k, r in LEXRE.items():
        m = (BACKRE if k == "back" else r).search(txt)
        if m and m.start() < best[0]: best = (m.start(), k)
    return best[1]

def load(name):
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    return tok, m

def chat(tok, msgs, prefill=""):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill

@torch.no_grad()
def last_logprobs(tok, m, texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], return_tensors="pt", padding=True).to(DEV)
        lg = m(**b).logits[:, -1].float()
        out.append(torch.log_softmax(lg, -1).cpu())
    return torch.cat(out)

def tid(tok, w):
    ids = tok(w, add_special_tokens=False).input_ids
    return ids[0]

@torch.no_grad()
def hidden(tok, m, texts, pos_fn, layers, bs=16):
    """pos_fn(text, ids_list) -> list of positions to average (indices in unpadded ids). Returns [N, len(layers), d]."""
    res = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        b = tok(chunk, return_tensors="pt", padding=True).to(DEV)
        hs = m(**b, output_hidden_states=True).hidden_states
        L = b.input_ids.shape[1]
        for j, t in enumerate(chunk):
            ids = tok(t).input_ids; off = L - len(ids)
            ps = [off + p for p in pos_fn(t, ids)]
            res.append(torch.stack([hs[l][j, ps].float().mean(0) for l in layers]).cpu().numpy())
    return np.stack(res)
