set -x
ulimit -n 65536 || true; cd "$2"; export OUTDIR="$1"; mkdir -p "$OUTDIR"
(uv pip install -q --python "$(command -v python)" mujoco scikit-learn || (python -m ensurepip >/dev/null && python -m pip install -q mujoco scikit-learn)) 2>&1 | tail -2
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u sj_collect.py 2>&1 | grep --line-buffered -v -i "warn\|Loading"
