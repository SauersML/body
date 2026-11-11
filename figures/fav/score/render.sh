#!/bin/zsh
# usage: render.sh NAME  (renders forearm.song -> NAME.wav, writes NAME.done)
cd ~/patina && ./target/release/patina --play $BODY_REPO/figures/fav/score/forearm.song --render $BODY_REPO/figures/fav/score/$1.wav > $BODY_REPO/figures/fav/score/$1.log 2>&1; touch $BODY_REPO/figures/fav/score/$1.done
