s = open("run5_head.py").read()
s = s.replace('"""body-causal run 2:', '"""body-causal run 5 (stem interventions): edits only at the QUESTION-STEM tokens (before the answer options), never at the options or the answer, so any change must travel through what the model computed from the scene.\nOld run-2 doc:')
s = s.replace("mask[r, len(penc[j]) - 1:len(cenc[j])] = True", "mask[r, STEM[j][0]:STEM[j][1]] = True")
s = s.replace("order = np.argsort(", '''def stem_cut(q, it):
    if it["a"] in ("yes", "no"): return len(q)
    c = [q.rfind(" " + w) for w in {it["a"], it["b"], it["a"].capitalize(), it["b"].capitalize()}]
    c = [x for x in c if x > 0]; return min(c) if c else len(q)
def stem_span(k):
    it = ITEMS[k]; u = it["q"] + suffix(it); p = chat(u); s0 = p.find(u)
    return (len(tok(p[:s0], add_special_tokens=False).input_ids), len(tok(p[:s0 + stem_cut(it["q"], it)], add_special_tokens=False).input_ids))
SPAN = [stem_span(k) for k in range(len(ITEMS))]; STEM = [SPAN[s[0]] for s in SEQ]
order = np.argsort(''', 1)
s = s.replace('m = torch.zeros_like(ids, dtype=torch.bool); m[:, -1] = True; MASK["m"] = m',
 'm = torch.zeros_like(ids, dtype=torch.bool)\n        for r, k in enumerate(ch): pad = T - int(enc.attention_mask[r].sum()); m[r, pad + SPAN[k][0]:pad + SPAN[k][1]] = True\n        MASK["m"] = m')
s = s.replace('MASK["m"] = torch.ones_like(nx, dtype=torch.bool)', 'MASK["m"] = torch.zeros_like(nx, dtype=torch.bool)')
s = s.replace('elif op[0] == "add": new = new + op[1]', 'elif op[0] == "add": new = new + op[1]\n            elif op[0] == "refl": u, mid = op[1], op[2]; new = new - 2 * ((new @ u) - mid)[..., None] * u')
assert "STEM[j][0]" in s and "pad + SPAN" in s and '"refl"' in s and "torch.zeros_like(nx" in s
s += open("run6_tail.py").read()
open("run6.py", "w").write(s)
