#!/bin/zsh
# usage: go3.sh ARM SEED STEPS -> v3/q06_ARM_sSEED.out
cd ${0:a:h}; export PYTORCH_MPS_HIGH_WATERMARK_RATIO=.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=.25 HF_HUB_OFFLINE=1 NE=${NE:-100} NC=${NC:-30}
exec ../.venv/bin/python v3.py $1 $2 $3 > v3/q06_$1_s$2.out 2>&1
