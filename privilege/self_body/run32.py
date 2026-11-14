import subprocess, os
env = dict(os.environ, RP_PARALLEL="1", RP_PYENV="cpcirc", RP_HF_MODELS="Qwen/Qwen3-8B Qwen/Qwen3-32B")
subprocess.Popen(["$BODY_WORK/islands/shell/hypha-rp-run", "hypha-prod-selfbody32", "A100 SXM,A100 PCIe,H100 PCIe,H100 SXM,H100 NVL,RTX PRO 6000,H200 SXM", "3", "--", "bash", "$BODY_RUNS.sh",
  "$BODY_RUNS/out", "Qwen/Qwen3-32B", "q32", "$BODY_RUNS.py", "$BODY_RUNS.py"], stdout=open("run32.log", "w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)
