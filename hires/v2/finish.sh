while pgrep -f "run8b3.sh|run8b4.sh" > /dev/null; do sleep 20; done
for p in long_person_rp short_person; do [ -f red8b_$p.npz ] && ../../.venv/bin/python ana_run.py red8b_$p.npz m8b_$p.json > /dev/null 2>&1; done
../../.venv/bin/python table.py > prefix_table.txt
