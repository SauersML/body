cd $BODY_REPO/figures/fav
for pass in 1 2 3 4; do
python3 -c "
import os;s=set(int(f[:5]) for f in os.listdir('fr'));m=[k for k in range(2040) if k not in s]
ch=[];
for k in m:
  if ch and k==ch[-1][1] and k-ch[-1][0]<15: ch[-1][1]=k+1
  else: ch.append([k,k+1])
print('\n'.join(f'{a}:{b}' for a,b in ch))" > todo.txt
[ -s todo.txt ] || break
cat todo.txt | xargs -P 6 -I{} timeout 150 python3 shoot.py video fr {} > /dev/null 2>&1
done
echo DONE > fill.done
