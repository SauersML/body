import subprocess, os, sys
tag, model, gpu, hrs = sys.argv[1:5]
env = dict(os.environ, RP_PARALLEL="1", RP_NO_SWEEP="1", RP_PYENV="cpcirc", RP_HF_MODELS=model)
name = f"hypha-prod-selfbody{tag}"
subprocess.Popen(["$BODY_WORK/islands/shell/hypha-rp-run", name, gpu, hrs, "--", "bash", "$BODY_WORK/selfbody12/go.sh",
  f"$BODY_WORK/runpod/{name}/out", model, tag, "$BODY_WORK/selfbody12/sb.py", "$BODY_WORK/selfbody12/bodies.py"],
  stdout=open(f"run{tag}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)
