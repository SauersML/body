"""Corpus baselines: GloVe (Wikipedia+Gigaword) and Qwen3-8B input embeddings. Project part words on word-list axes; bootstrap over axis words."""
import gzip, json, numpy as np, sys
sys.path.insert(0, "$BODY_WORK/foot")
PARTS = ["feet", "hands", "back", "shoulders", "neck", "knees", "belly", "face"]
SENS = "romantic sensual intimate seductive passionate erotic flirtatious desire lover kiss candlelit sultry tender caress honeymoon affectionate".split()
CLIN = "clinical practical routine procedural administrative office meeting invoice schedule paperwork inspection standard formal businesslike memo spreadsheet".split()
PLE = "pleasant lovely delightful wonderful nice gentle comforting soothing enjoyable joyful happy sweet calm warm beautiful cozy".split()
UNP = "unpleasant awful horrible nasty painful harsh irritating disgusting miserable sour bitter cold ugly gross annoying stressful".split()
need = set(PARTS + SENS + CLIN + PLE + UNP)
def glove():
    E = {}
    with gzip.open("$BODY_WORK/foot/glove100.gz", "rt") as f:
        for line in f:
            w, *v = line.rstrip().split(" ")
            if w in need and len(v) == 100: E[w] = np.array(v, np.float32)
    return E
def qwen():
    from safetensors import safe_open; import glob, json as J
    d = glob.glob("$BODY_WORK/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/")[0]
    idx = J.load(open(d + "model.safetensors.index.json"))["weight_map"]["model.embed_tokens.weight"]
    from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    with safe_open(d + idx, "pt") as f: W = f.get_tensor("model.embed_tokens.weight").float().numpy()
    E = {}
    for w in need:
        ids = tok(" " + w, add_special_tokens=False).input_ids
        if len(ids) == 1: E[w] = W[ids[0]]
    return E
def stat(E, A, B, rng=None):
    A = [w for w in A if w in E]; B = [w for w in B if w in E]
    if rng is not None: A = list(rng.choice(A, len(A))); B = list(rng.choice(B, len(B)))
    U = {w: E[w] / np.linalg.norm(E[w]) for w in E}
    ax = np.mean([U[w] for w in A], 0) - np.mean([U[w] for w in B], 0); ax /= np.linalg.norm(ax)
    p = np.array([U[w] @ ax for w in PARTS])
    return p, (p[0] - p[1:].mean()) / p[1:].std()
out = {}
for name, E in [("glove", glove()), ("qwen_inputemb", qwen())]:
    rng = np.random.default_rng(0)
    for an, A, B in [("sensual", SENS, CLIN), ("valence", PLE, UNP)]:
        p, z = stat(E, A, B)
        boot = [stat(E, A, B, rng)[1] for _ in range(2000)]
        out[f"{name}_{an}"] = dict(proj=dict(zip(PARTS, p.tolist())), z=float(z), ci=np.percentile(boot, [2.5, 97.5]).tolist(),
                                   rank=int((p > p[0]).sum()) + 1, missing=[w for w in A + B if w not in E])
        print(name, an, round(float(z), 2), np.round(np.percentile(boot, [2.5, 97.5]), 2), "rank", out[f"{name}_{an}"]["rank"], dict(zip(PARTS, np.round(p, 3))))
json.dump(out, open("$BODY_WORK/foot/res_corpus.json", "w"), indent=1)
