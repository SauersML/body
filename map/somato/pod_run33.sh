cd /workspace/body; export HF_HOME=/workspace/hf
until [ -f done32.txt ]; do sleep 15; done
mkdir -p out3_32b
python extract3.py Qwen/Qwen3-32B out3_32b out2_32b/prefix.json > ex33.log 2>&1
echo ALLDONE > done33.txt
