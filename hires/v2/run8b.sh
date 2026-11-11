for p in long short none; do
  f=prefix_$p.json; [ $p = none ] && f=none
  ../../.venv/bin/python extract2.py Qwen/Qwen3-8B $f out8b_$p 33 "[16,18,20,22,23,24,25,26,27,28,29,30,31,32]" > ex_$p.log 2>&1
done
