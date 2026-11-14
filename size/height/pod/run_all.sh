#!/bin/bash
# run_all.sh OUT RUNPY : every Qwen3 model, small to large; deletes each model's weights after.
O=$1; RUN=$2; mkdir -p $O; export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $O/gpu.txt
for M in ${MODELS:-Qwen3-0.6B Qwen3-0.6B-Base Qwen3-1.7B Qwen3-1.7B-Base Qwen3-4B Qwen3-4B-Base Qwen3-8B Qwen3-8B-Base Qwen3-14B Qwen3-14B-Base Qwen3-32B}; do
  [ -f $O/$M/res.json ] && continue
  python -c "from huggingface_hub import snapshot_download as s; s('Qwen/$M', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])" > $O/dl_$M.log 2>&1
  python $RUN Qwen/$M $O/$M 2>&1 | grep -v -i "warn\|Loading weights" > $O/$M.log
  rm -rf $HF_HOME/hub/models--Qwen--$M $HF_HOME/xet
done
echo ALLDONE > $O/DONE
