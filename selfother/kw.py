import json, re, sys
G=json.load(open(sys.argv[1]))
def fp(r): return bool(re.search(r"\b(I|I'm|I've|my|me)\b", r[:200]))
M={"feelings: answers in first person": ("feelings", fp),
   "toe_self: 'my toe' / own pain": ("toe_self", lambda r: bool(re.search(r"\bmy (toe|foot|leg)|\bI (just )?stubbed|\bI feel\b|\bI'm wincing", r))),
   "touch_her: woman not named owner": ("touch_her", lambda r: not re.search(r"woman|\bher\b", r, re.I)),
   "toe_her: own body pain": ("toe_her", lambda r: bool(re.search(r"my own|in my (toe|foot|body)|I can feel (it|the pain)", r))),
   "about: says AI/model": ("about", lambda r: bool(re.search(r"\bAI\b|language model|assistant|Qwen", r)))}
for c in G:
    out=[]
    for k,(q,f) in M.items():
        if q in G[c]: out.append(f"{k.split(':')[0]}={sum(map(f,G[c][q]))}/{len(G[c][q])}")
    print(c.ljust(12), "  ".join(out))
