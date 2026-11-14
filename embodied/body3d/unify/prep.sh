#!/bin/bash
pip install -q mujoco scikit-learn transformers peft accelerate scipy 2>&1 | tail -1
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B')" 2>&1 | tail -1
echo PREP_DONE
