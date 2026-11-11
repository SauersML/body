import torch, glob, os, gc
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors import safe_open
from huggingface_hub import snapshot_download
def load(M="Qwen/Qwen3-8B"):
    d = snapshot_download(M, local_files_only=True); cfg = AutoConfig.from_pretrained(M)
    with torch.device("meta"): m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    for f in sorted(glob.glob(d + "/*.safetensors")):
        sd = {}
        with safe_open(f, framework="pt", device="cpu") as fh:
            for k in fh.keys(): sd[k] = fh.get_tensor(k).to("mps"); 
        m.load_state_dict(sd, strict=False, assign=True); del sd; gc.collect(); print(f, torch.mps.current_allocated_memory()//2**30, flush=True)
    if cfg.tie_word_embeddings: m.lm_head.weight = m.model.embed_tokens.weight
    m.model.rotary_emb = type(m.model.rotary_emb)(config=cfg).to("mps")
    return m.eval()
if __name__ == "__main__":
    m = load(); print("ok", flush=True); print(m(torch.tensor([[1,2,3]], device="mps")).logits[0,-1,:3])
