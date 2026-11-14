#!/bin/bash
# steer.sh MODEL DELTA  (on box after b3s finished: final ckpt in /root/out)
cd /root/c8; export OUTDIR=/root/out CKDIR=/root/out; T=$(basename $1)
export HF_HUB_ENABLE_HF_TRANSFER=0 X2_CANON=1 X2_GSMOOTH=1 X2_GCHI=.07 X2_GCLO=.04 DELTA=$2
cp /root/out/final_${T}_named_dagger.pt /root/out/ckpt_${T}_named.pt
DO=rep NP=400 python -u steer_confirm.py $1 named 2>&1 | grep --line-buffered -v -i "warn\|Loading"
DO= python -u press3.py $1 named 2>&1 | grep --line-buffered -v -i "warn\|Loading"
echo STEERSH_EXIT
