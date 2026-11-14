#!/bin/bash
# runu.sh ARM  (on vast box; code in /root/uni, ckpt /root/ck)
cd /root/uni; export OUTDIR=/root/out; mkdir -p $OUTDIR; ulimit -n 65536 || true
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 NW=${NW:-24} INIT_CK=/root/ck/x8abs_final_Qwen3-8B_named.pt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u unify.py $1 2>&1 | grep --line-buffered -v -i "warn\|Loading"
echo RUNU_EXIT
