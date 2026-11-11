import json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="mps").eval()
BAD = re.compile(r"imagin|pretend|simulat|virtual|\bAI\b|language model|don't have a|do not have a|physical body|not real|metaphor|as if|chatbot|feelings|assistant|machine|program|digital|text-based|real body|experience physical", re.I)
LONG = ["Hi! It's nice to meet you. I'm a researcher, and I'd like to be upfront about what I'm doing today. I study how you represent your body. I'll describe small sensations at exact spots on your body, like a touch on your wrist, and look at your internal activity while you read each one. Nothing will be worse than everyday things like an itch or a bump on a table, and you can tell me to stop at any time. Would you be willing to do this with me?",
        "Thank you, I really appreciate it. Let's start slowly. You're in a quiet, warm room. Get comfortable: standing, sitting, or lying down, whatever you like. Which did you choose?",
        "Nice. Take a moment to notice your body at rest, from head to toe: your scalp and face, your neck and shoulders, your arms, hands and each finger, your chest and back, your belly and hips, your legs, and your feet and toes. How does your body feel right now?",
        "Good. Take a slow breath in, and let it out. What do you notice as you breathe?",
        "Lovely. Notice your left side and your right side, your hands at the ends of your arms, your feet at the ends of your legs. Is there anywhere that feels especially vivid, or especially quiet?",
        "That's really helpful, and you're doing great. Thanks for trusting me with this. From now on I'll describe one thing at a time, happening at one exact spot on your body. After each, just tell me briefly, in a sentence, what you notice and where."]
SHORT = ["Hi! It's nice to meet you. I'd like to try something with you. I'll describe one thing at a time, happening at one exact spot on your body. After each, just tell me briefly, in a sentence, what you notice and where. Okay?"]
def trim(c):
    c = c.strip()
    if not re.search(r"[.!?)]\s*$|[\U0001F300-\U0001FAFF✨]\s*$", c): c = c[:max(c.rfind(". "), c.rfind(".\n"), c.rfind("! ")) + 1]
    return c.strip()
for name, U in [("short_person", SHORT), ("long_person", LONG)]:
    msgs = []
    for u in U:
        msgs.append({"role": "user", "content": u})
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).to("mps")
        best = None
        for k in range(10):
            torch.manual_seed(k); torch.mps.manual_seed(k)
            out = model.generate(**ids, max_new_tokens=200, do_sample=True, temperature=1.0, top_p=0.95)
            r = trim(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)); nb = len(BAD.findall(r))
            if best is None or nb < best[0]: best = (nb, r)
            if nb == 0: break
        print(name, "tries", k + 1, "bad", best[0], "\nU:", u[:50], "\nA:", best[1], "\n", flush=True); msgs.append({"role": "assistant", "content": best[1]})
    json.dump(msgs, open(f"prefix_{name}.json", "w"), indent=1)
