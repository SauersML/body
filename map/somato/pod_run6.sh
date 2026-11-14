cd /workspace/body; export HF_HOME=/workspace/hf
until [ -f done5.txt ]; do sleep 10; done
python steer4.py Qwen/Qwen3-32B out3_32b out2_32b 52 > st42.log 2>&1
echo ALLDONE > done6.txt
