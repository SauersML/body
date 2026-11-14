cd /workspace/body; export HF_HOME=/workspace/hf
python steer3.py Qwen/Qwen3-32B out2_32b 52 > st32.log 2>&1
echo ALLDONE > done5.txt
