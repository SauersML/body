export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/dev/shm/hf
huggingface-cli download Qwen/Qwen3-32B >/dev/null 2>&1
while pgrep -f "run.py Qwen/Qwen3-8B" >/dev/null; do sleep 10; done
python run.py Qwen/Qwen3-32B main q32 > log32.txt 2>&1
echo ALLDONE >> log32.txt
