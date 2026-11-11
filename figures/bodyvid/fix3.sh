cd $BODY_REPO/figures/bodyvid
O=$BODY_REPO/figures/out/body_video_smooth
for k in 0 1 2 3 4 5 6 7; do a=$((642+k*107)); b=$((a+107)); OMP_NUM_THREADS=2 python3 v3.py seg $a $b $O.fix$k.mp4 & done; wait
: > $O.list
for p in 00 01 02; do echo "file '$O.part$p.mp4'" >> $O.list; done
for k in 0 1 2 3 4 5 6 7; do echo "file '$O.fix$k.mp4'" >> $O.list; done
for p in 07 08; do echo "file '$O.part$p.mp4'" >> $O.list; done
ffmpeg -y -loglevel error -f concat -safe 0 -i $O.list -c copy $O.v.mp4 && echo DONE
