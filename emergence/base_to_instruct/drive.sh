#!/bin/bash
# runs every line of list.txt; prefetches the next checkpoint while the current one runs; deletes each after use
cd /root/em; export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/root/hf
fetch(){ huggingface-cli download $1 --revision $2 --include "*.json" "*.safetensors" "*.txt" "*.model" "*.py" >/dev/null 2>&1; }
mapfile -t LINES < $1
read m r t c <<< "${LINES[0]}"; fetch $m $r
for i in "${!LINES[@]}"; do
  read m r t c <<< "${LINES[$i]}"
  nx="${LINES[$((i+1))]}"; if [ -n "$nx" ]; then read m2 r2 t2 c2 <<< "$nx"; fetch $m2 $r2 & PF=$!; fi
  [ -f out/$t.json ] || python emerge.py $m $r $t $c >> log.txt 2>&1
  [ -n "$nx" ] && wait $PF
  # delete this revision's blobs (keep next)
  python - "$m" "$r" "$m2" "$r2" <<'PY'
import sys; from huggingface_hub import scan_cache_dir
m,r,m2,r2=sys.argv[1:5]; c=scan_cache_dir()
for repo in c.repos:
  if repo.repo_id!=m: continue
  keep={x.commit_hash for x in repo.revisions if (m==m2 and r2 in x.refs)}
  hs=[x.commit_hash for x in repo.revisions if r in x.refs and x.commit_hash not in keep]
  if hs: c.delete_revisions(*hs).execute()
PY
done
echo ALLDONE >> log.txt
