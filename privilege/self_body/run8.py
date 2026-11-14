import subprocess, os
env = dict(os.environ, RP_PARALLEL="1", RP_PYENV="cpcirc", RP_HF_MODELS="Qwen/Qwen3-8B")
subprocess.Popen(["$BODY_WORK/islands/shell/hypha-rp-run", "hypha-prod-selfbody8", "48gb", "2.5", "--", "bash", "$BODY_RUNS.sh",
  "$BODY_RUNS/out", "Qwen/Qwen3-8B", "q8", "$BODY_RUNS.py", "$BODY_RUNS.py"], stdout=open("run8.log", "w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)
