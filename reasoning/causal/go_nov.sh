#!/bin/bash
cd /workspace/bc
export HF_HOME=/workspace/hf OMP_NUM_THREADS=13 PYTHONUNBUFFERED=1
pip install -q 'transformers==4.51.3' accelerate > pip.log 2>&1
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B')" > dl.log 2>&1 &
SMOKE=1 python nov_run.py outsmoke Qwen/Qwen3-0.6B base,states,dirs,das,attr,patch > smoke.log 2>&1 || { echo SMOKEFAIL >> smoke.log; exit 1; }
wait
python nov_run.py out32 Qwen/Qwen3-32B base,states,attr,patch > run_a.log 2>&1
python nov_run.py out32 Qwen/Qwen3-32B states,dirs > run_b.log 2>&1
python nov_run.py out32 Qwen/Qwen3-32B states,das > run_c.log 2>&1
echo ALLDONE > done.txt
