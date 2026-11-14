#!/bin/zsh
cd ${0:a:h}
while [ ! -f train2_log.json ]; do sleep 20; done
./run.sh 3
for T in 2 3; do until [ -f interp$T.json ]; do ~/.venv/bin/python interp2.py $T >> interp$T.out 2>&1; sleep 5; done; done
~/.venv/bin/python figs2.py > figs2.out 2>&1
