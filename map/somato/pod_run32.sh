cd /workspace/body; export HF_HOME=/workspace/hf
while pgrep -f "python steer" >/dev/null; do sleep 15; done
mkdir -p out2_32b
python extract2.py Qwen/Qwen3-32B out2_32b > ex32.log 2>&1
python steer3.py Qwen/Qwen3-32B out2_32b 44 > st32.log 2>&1
echo ALLDONE > done32.txt
