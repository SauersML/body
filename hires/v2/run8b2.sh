K="[16,18,20,22,23,24,25,26,27,28,29,30,31,32]"
for p in long_person short_person; do
  ../../.venv/bin/python extract2.py Qwen/Qwen3-8B prefix_$p.json out8b_$p 33 "$K" > ex_$p.log 2>&1 && ../../.venv/bin/python reduce.py out8b_$p red8b_$p.npz > /dev/null 2>&1
  STYLE=narrated ../../.venv/bin/python extract2.py Qwen/Qwen3-8B prefix_$p.json out8b_${p}_rp 33 "$K" > ex_${p}_rp.log 2>&1 && ../../.venv/bin/python reduce.py out8b_${p}_rp red8b_${p}_rp.npz > /dev/null 2>&1
done
