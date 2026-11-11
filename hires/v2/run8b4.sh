sleep 30; while pgrep -f "run8b2.sh|run8b3.sh" > /dev/null; do sleep 20; done
export BS=16; K="[16,18,20,22,23,24,25,26,27,28,29,30,31,32]"
[ -f out8b_short_person_rp/rows.json ] && ../../.venv/bin/python reduce.py out8b_short_person_rp red8b_short_person_rp.npz > /dev/null 2>&1
../../.venv/bin/python extract2.py Qwen/Qwen3-8B prefix_short_person.json out8b_short_person 33 "$K" > ex_short_person.log 2>&1 && ../../.venv/bin/python reduce.py out8b_short_person red8b_short_person.npz > /dev/null 2>&1
