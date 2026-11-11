import numpy as np
from ana2 import ST, VE, READ, KEYS, placedirs, TOE, FOOT, HANDF, MAP
from ana3 import H, cosp, z
for pos in ["dp","reply12","reply"]:
  for L in [25,28,32]:
    Pk,mu=placedirs(L)
    for nm,(c,q,b) in {"OWN":("normal","self_toe","self_burn"),"woman":("normal","woman_toe","friend_burn"),"woman+me":("me","woman_toe","friend_burn")}.items():
        d=H(c,q,pos,L)-H(c,b,pos,L); p=cosp(d,Pk); o=np.argsort(-p)
        print(pos,L,nm,"top",[KEYS[i] for i in o[:6]],"bottom",[KEYS[i] for i in o[-6:]])
