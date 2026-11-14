#!/bin/bash
# v2 pipeline on the pod. cwd = /workspace/body (downloads started separately: dl8.ok, dl32.ok, pip.log PIPOK)
cd /workspace/body
export HF_HOME=/workspace/hf
until grep -q PIPOK pip2.log && [ -f dl8.ok ]; do sleep 10; done
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name())"
mkdir -p out2_8b; cp out_8b/prefix.json out2_8b/
python extract2.py Qwen/Qwen3-8B out2_8b > ex8.log 2>&1
python steer3.py Qwen/Qwen3-8B out2_8b 25 > st8.log 2>&1
mkdir -p out2_32b
until [ -f dl32.ok ]; do sleep 10; done
python extract2.py Qwen/Qwen3-32B out2_32b > ex32.log 2>&1
python steer3.py Qwen/Qwen3-32B out2_32b 44 > st32.log 2>&1
echo ALLDONE > done.txt
