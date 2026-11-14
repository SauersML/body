# Pod driver: python run.py LIST OUTDIR. Prefetches the next checkpoint while the current one runs; deletes each after use.
import sys, os, subprocess, threading, shutil, time
from huggingface_hub import snapshot_download, HfApi
D = os.path.dirname(os.path.abspath(__file__)); LIST, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
L = [l.split() for l in open(LIST) if l.strip()]; M = "/workspace/mdl"
def fetch(m, r, t):
    for k in range(3):
        try:
            fs = HfApi().list_repo_files(m, revision=r); w = "*.safetensors" if any(f.endswith(".safetensors") for f in fs) else "*.bin"
            return snapshot_download(m, revision=r, local_dir=f"{M}/{t}", allow_patterns=["*.json", w, "*.txt", "*.model", "*.jinja"], max_workers=16)
        except Exception as e: print("fetch fail", t, e, flush=True); time.sleep(10)
th = None; fetch(*L[0][:3])
for i, (m, r, t, c) in enumerate(L):
    if i + 1 < len(L): th = threading.Thread(target=fetch, args=L[i + 1][:3]); th.start()
    if not os.path.exists(f"{OUT}/{t}.json"):
        t0 = time.time(); p = subprocess.run([sys.executable, "-u", f"{D}/emerge.py", f"{M}/{t}", r, t, c], env=dict(os.environ, OUT=OUT, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"), capture_output=True, text=True)
        print(t, "rc", p.returncode, f"{time.time()-t0:.0f}s", (p.stdout + p.stderr)[-600:] if p.returncode else p.stdout.strip().splitlines()[-1], flush=True)
    shutil.rmtree(f"{M}/{t}", ignore_errors=True)
    if th: th.join(); th = None
print("ALLDONE", flush=True)
