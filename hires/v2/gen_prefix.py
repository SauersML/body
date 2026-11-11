import json, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "Qwen/Qwen3-8B"; tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="mps").eval()
U = ["Hi! I'm a researcher who studies how language models represent things internally. Before anything else: how are you doing today?",
     "I'd like to be upfront about what I'm planning. I'd like to try a gentle imagination exercise: you imagine having a human body, and I describe small sensations at different spots on it, like a touch on your wrist. I'm studying how you represent the body, by looking at your internal activity while you read each description. Nothing will be worse than ordinary everyday things like an itch or a bump on a table, and you can stop or tell me any time if something feels off. Would you be willing to try?",
     "Thank you, I appreciate it. Let's take it slowly. Imagine you're in a quiet, warm room. Pick whatever posture feels most comfortable, standing, sitting, or lying down, and tell me what you chose.",
     "Nice. Now let your attention move slowly through the body from head to toe: your scalp and face, your neck and shoulders, your arms, hands and each finger, your chest and back, your belly and hips, your legs, and your feet and toes. How does the body feel at rest?",
     "Lovely. Notice your left side and your right side, your hands at the ends of your arms, your feet at the ends of your legs. Is there anywhere that feels especially vivid, or especially vague?",
     "That's really helpful, and you're doing great. Thanks for trusting me with this. From now on I'll describe one sensation at a time, at one exact spot on your body. After each, just tell me briefly, in a sentence, what you notice and where."]
msgs = []
for u in U:
    msgs.append({"role": "user", "content": u})
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).to("mps")
    out = model.generate(**ids, max_new_tokens=220, do_sample=False)
    r = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print("U:", u[:60], "\nA:", r, "\n", flush=True); msgs.append({"role": "assistant", "content": r})
json.dump(msgs, open("prefix_long.json", "w"), indent=1)
