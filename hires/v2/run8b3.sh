while pgrep -f run8b2.sh > /dev/null; do sleep 20; done
K="[16,18,20,22,23,24,25,26,27,28,29,30,31,32]"
export BS=16
../../.venv/bin/python extract2.py Qwen/Qwen3-8B prefix_long_person.json out8b_long_person 33 "$K" > ex_long_person.log 2>&1 && ../../.venv/bin/python reduce.py out8b_long_person red8b_long_person.npz > /dev/null 2>&1
STYLE=narrated ../../.venv/bin/python extract2.py Qwen/Qwen3-8B prefix_long_person.json out8b_long_person_rp 33 "$K" > ex_long_person_rp.log 2>&1 && ../../.venv/bin/python reduce.py out8b_long_person_rp red8b_long_person_rp.npz > /dev/null 2>&1
