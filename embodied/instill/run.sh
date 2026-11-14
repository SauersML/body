#!/bin/zsh
# usage: run.sh TAG ; loops until the run's log exists (survives external kills)
cd ${0:a:h}; TAG=$1; [ $TAG = 3 ] && export NAMELESS=1
until [ -f train${TAG}_log.json ]; do ~/.venv/bin/python bi2train.py 2000 >> train${TAG}.out 2>&1; sleep 5; done
