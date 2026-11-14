#!/bin/bash
# run.sh MODEL CKPT DELTA  (on vast box, in /root/c8)
cd /root/c8; export OUTDIR=/root/out; mkdir -p $OUTDIR; ulimit -n 65536 || true
pip install -q mujoco scikit-learn transformers peft accelerate hf_transfer scipy 2>&1 | tail -2
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 X2_CANON=1 X2_GSMOOTH=1 X2_GCHI=.07 X2_GCLO=.04 STEPS1=2000 ROUNDS=4 STEPS_R=600 INIT_CK=/root/c8ck/$2 DELTA=$3
export NW=${NW:-24}
nproc; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u b3s.py $1 2>&1 | grep --line-buffered -v -i "warn\|Loading"
echo RUNSH_EXIT
