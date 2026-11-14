export HF_HOME=/dev/shm/hf HF_HUB_OFFLINE=1
python run2.py Qwen/Qwen3-8B q8 11,19,27 > log2_8.txt 2>&1
python run2.py Qwen/Qwen3-32B q32 19,31,43 > log2_32.txt 2>&1
echo ALLDONE >> log2_32.txt
