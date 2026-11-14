cd /workspace/body; export HF_HOME=/workspace/hf
until [ -f done.txt ]; do sleep 15; done
mkdir -p out3_8b
python extract3.py Qwen/Qwen3-8B out3_8b out_8b/prefix.json > ex3.log 2>&1
python steer4.py Qwen/Qwen3-8B out3_8b out2_8b 25 > st4.log 2>&1
echo ALLDONE > done3.txt
