import json, torch, sys; import stim_c as C
from transformers import AutoTokenizer, AutoModelForCausalLM
M = sys.argv[1]; tok = AutoTokenizer.from_pretrained(M); dev = "mps" if torch.backends.mps.is_available() else "cuda"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
msgs = json.load(open(sys.argv[2])); I = C.items(); res = []
for it in I:
    if not (it["agent"] == "water" and it["path"] in ("arm_palm_left", "leg_left", "head") and it["kind"] == "body"): continue
    for i in [2, 8, 16]:
        t = it["sent"] + "".join(f" … {it['step']}" for _ in range(i))
        for q in ["", " Where on your body is the drop right now?"]:
            s = tok.apply_chat_template(msgs + [{"role": "user", "content": t + q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            with torch.no_grad(): o = model.generate(ids, max_new_tokens=40, do_sample=False)
            a = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True); res.append(dict(path=it["path"], dir=it["dir"], i=i, q=bool(q), a=a)); print(it["path"], it["dir"], i, bool(q), "|", a.replace("\n", " "), flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
