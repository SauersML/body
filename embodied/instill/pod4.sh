set -x
# usage: pod4.sh OUTDIR CODEDIR MODEL ARM SEED STEPS [interp]
ulimit -n 65536 || true; cd "$2"; export OUTDIR="$1"; mkdir -p "$OUTDIR"; M=$3; A=$4; S=$5; N=$6
python -c "import scipy, peft, transformers; print(scipy.__version__, peft.__version__, transformers.__version__)" || pip install -q scipy
nproc; nvidia-smi --query-gpu=name,memory.total --format=csv
python -u v4.py $A $S $N $M 2>&1 | grep --line-buffered -v -i "warn\|Loading"
TAG="$(basename $M)_${A}_s${S}"
if [ -f interp4.py ] && [ "$7" = "interp" ]; then python -u interp4.py $TAG $A $M 2>&1 | grep --line-buffered -v -i "warn\|Loading"; fi
