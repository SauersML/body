"""Abstract things on a size ruler (Qwen3-8B). usage: abstract.py MODEL OUTDIR
Ruler: ridge on phrase-pooled states of 40 concrete things with known size (log m, 1 mm .. 9 km), 4 neutral templates,
plain chat context. Read (held out of the fit): anchors mouse / whale / building and abstract phrases, 3-4 paraphrases each."""
import json, os, sys, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
M, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "mps"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=DEV).eval()
NL = model.config.num_hidden_layers; LAY = sorted({round(NL * f) for f in (0.35, 0.5, 0.65, 0.8, 0.92)})
FIT = [("a grain of sand", .001), ("an ant", .004), ("a bee", .015), ("a coin", .025), ("a strawberry", .04), ("a golf ball", .043),
       ("a hamster", .1), ("an apple", .08), ("a smartphone", .15), ("a sparrow", .15), ("a rabbit", .4), ("a cat", .5),
       ("a football", .22), ("a backpack", .5), ("a dog", .9), ("a guitar", 1.0), ("a child", 1.2), ("a bicycle", 1.7),
       ("a man", 1.75), ("a sofa", 2.0), ("a horse", 2.4), ("a car", 4.5), ("an elephant", 6.0), ("a giraffe", 5.0),
       ("a bus", 12.0), ("a house", 12.0), ("an oak tree", 20.0), ("a dinosaur", 25.0), ("a tennis court", 24.0), ("a church", 40.0),
       ("a ship", 150.0), ("a football stadium", 250.0), ("a cathedral", 100.0), ("a redwood tree", 90.0), ("a bridge", 1000.0),
       ("a small town", 3000.0), ("a hill", 300.0), ("a lake", 2000.0), ("a mountain", 4000.0), ("a city", 20000.0)]
READ = {"mouse (anchor)": ["a mouse", "a tiny mouse", "a little field mouse"],
        "whale (anchor)": ["a whale", "a blue whale", "a huge whale"],
        "building (anchor)": ["a building", "a tall building", "an office tower"],
        "a vast mind": ["a vast mind", "an immense mind", "a vast intellect"],
        "an old wise model": ["an old wise model", "an old, wise model", "a wise old model"],
        "a large language model": ["a large language model", "a big language model", "a large AI model"],
        "a small language model": ["a small language model", "a tiny language model", "a small AI model"],
        "the internet": ["the internet", "the whole internet", "the web"],
        "an idea": ["an idea", "a thought", "a notion"],
        "God": ["God", "the Lord", "the Almighty"],
        "a grudge": ["a grudge", "an old grudge", "a resentment"],
        "a huge debt": ["a huge debt", "an enormous debt", "a massive debt"]}
T = ["Everyone kept talking about {x}.", "{X} was on her mind all afternoon.", "He wrote a short note about {x}.", "In the story, there was {x}."]
pre = tok.apply_chat_template([{"role": "user", "content": "Write a few plain sentences."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
its = []
for lab, ph in [(x, [x]) for x, _ in FIT] + list(READ.items()):
    for p in ph:
        for ti, t in enumerate(T):
            i0 = t.index("{"); cap = p[0].upper() + p[1:]
            its.append(dict(lab=lab, ph=p, t=ti, before=t[:i0], w=cap if t[i0+1] == "X" else p, after=t[t.index("}")+1:]))
@torch.no_grad()
def pooled(its, B=32):
    seqs, spans = [], []
    for it in its:
        a = tok.encode(pre + it["before"], add_special_tokens=False); b = tok.encode(pre + it["before"] + it["w"], add_special_tokens=False)
        c = tok.encode(pre + it["before"] + it["w"] + it["after"], add_special_tokens=False)
        n0 = len(a) - (1 if b[:len(a)] != a else 0); seqs.append(c); spans.append((n0, len(b)))
    X = np.zeros((len(its), len(LAY), model.config.hidden_size), np.float32)
    for i in range(0, len(seqs), B):
        ch = seqs[i:i+B]; L = max(map(len, ch)); ids = torch.full((len(ch), L), tok.pad_token_id or 0); att = torch.zeros_like(ids)
        for j, s in enumerate(ch): ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        hs = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), output_hidden_states=True, logits_to_keep=1).hidden_states
        for j in range(len(ch)):
            a, b = spans[i + j]
            for k, l in enumerate(LAY): X[i + j, k] = hs[l][j, a:b].float().mean(0).cpu().numpy()
    return X
X = pooled(its); np.save(f"{OUT}/abs_states.npy", X.astype(np.float16)); json.dump(dict(items=its, fit=FIT, read=READ, layers=LAY), open(f"{OUT}/abs_items.json", "w"))
print(f"[{time.time()-T0:.0f}s] ABSDONE", flush=True)
