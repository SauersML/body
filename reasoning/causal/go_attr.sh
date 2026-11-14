#!/bin/bash
cd /root/bc
export HF_HUB_ENABLE_HF_TRANSFER=1 OMP_NUM_THREADS=16
(huggingface-cli download Qwen/Qwen3-32B > dl32.log 2>&1 || hf download Qwen/Qwen3-32B > dl32.log 2>&1) &
python attr_run.py out8 Qwen/Qwen3-8B > run8.log 2>&1
wait
python attr_run.py out32 Qwen/Qwen3-32B > run32.log 2>&1
touch ALLDONE
