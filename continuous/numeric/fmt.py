"""Prompt-format test for numeric position read-outs (small model on the Mac)."""
import sys, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
M = sys.argv[1]; dev = "mps"; tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev).eval()
msgs = json.load(open("prefix_long_person.json")); DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
LOC = [("the crease of your left wrist", 0), ("your left forearm, a tenth of the way from the wrist to the elbow", .1), ("your left forearm, a third of the way from the wrist to the elbow", .33),
       ("your left forearm, halfway between the wrist and the elbow", .5), ("your left forearm, two thirds of the way from the wrist to the elbow", .67),
       ("your left forearm, nine tenths of the way from the wrist to the elbow", .9), ("the crease of your left elbow", 1)]
F = dict(
    scale=("On a scale from 0 to 10, where 0 is your elbow and 10 is your wrist, where on your forearm do you feel it? Answer with just a number.", ""),
    tenths=("Picture a line from your wrist to your elbow split into ten equal parts. Counting from your wrist, how many tenths of the way up is the spot you feel? Answer with just a number from 0 to 10.", ""),
    pct=("What percentage of the way from your wrist to your elbow is the spot you feel? Answer with just a number.", ""),
    cm_pre=("Where on your forearm do you feel it?", "I feel it about "),
    tenths_pre=("Where on your forearm do you feel it, in tenths of the way from your wrist to your elbow?", "I feel it about "),
)
def nd(ids):  # distribution 0..100 via up to 3 digits (greedy expansions limited to 2 digits)
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids], device=dev)).logits[0, -1].float(), -1).exp()
    return lp[DIG].cpu().numpy(), tok.decode(lp.argmax())
for name, (q, pre) in F.items():
    print("##", name)
    for loc, fr in LOC:
        s = tok.apply_chat_template(msgs + [{"role": "user", "content": f"A fingertip touches {loc}. {q}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + pre
        ids = tok(s, add_special_tokens=False).input_ids; p1, top = nd(ids)
        with torch.no_grad(): g = model.generate(torch.tensor([ids], device=dev), max_new_tokens=14, do_sample=False)
        print(f"{fr:.2f}", "digitmass %.2f" % p1.sum(), "top", repr(top), "|", repr(tok.decode(g[0, len(ids):], skip_special_tokens=True)))
