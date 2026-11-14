#!/bin/zsh
cd ${0:a:h}; export PYTORCH_MPS_HIGH_WATERMARK_RATIO=.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=.25 HF_HUB_OFFLINE=1 NE=100 NC=30
while pgrep -f "v3.py named 0 1500x" >/dev/null; do sleep 20; done
for R in "named 0" "nameless 0" "named 1" "nameless 1" "named 2" "nameless 2"; do
  A=${R% *}; S=${R#* }; TAG=Qwen3-0.6B_${A}_s${S}
  [ -f v3/train_$TAG.json ] || ../.venv/bin/python v3.py $A $S 2500 > v3/q06_${A}_s${S}.out 2>&1
  [ -f v3/interp_$TAG.json ] || ../.venv/bin/python interp3.py $TAG $A > v3/i06_${A}_s${S}.out 2>&1
done
