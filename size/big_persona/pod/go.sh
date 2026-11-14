#!/bin/bash
O=$1; mkdir -p $O; export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0
D=$(cd $(dirname $0); pwd)
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $O/gpu.txt
python -c "import transformers as t,sys; v=tuple(map(int,t.__version__.split('.')[:2])); sys.exit(0 if v>=(4,56) else 1)" || pip install -q "transformers==4.57.1" accelerate 2>&1 | tail -1
python -c "import accelerate" 2>/dev/null || pip install -q accelerate
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B', allow_patterns=['*.json','*.safetensors','*.txt'])" > $O/dl.log 2>&1
cp $3 $4 $D/ 2>/dev/null; python $2 $O 2>&1 | grep -v -i "warn\|Loading" > $O/run.log
