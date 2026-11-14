cd /workspace/body; export HF_HOME=/workspace/hf BS=4 GBS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python extract2.py Qwen/Qwen3-32B out2_32b > ex32.log 2>&1
python extract3.py Qwen/Qwen3-32B out3_32b out2_32b/prefix.json > ex33.log 2>&1
python steer3.py Qwen/Qwen3-32B out2_32b 44 > st32.log 2>&1
echo ALLDONE > done4.txt
