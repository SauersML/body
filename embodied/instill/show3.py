import json,sys,numpy as np
r=json.load(open(sys.argv[1]))
for k,v in r.items():
  if k in('steer','arcs'):continue
  if k=='pos_answer_curve': print(k,np.round(v,1).tolist());continue
  print(k, {a:round(b,2) for a,b in v.items()} if isinstance(v,dict) else round(v,3))
for m in r['steer']: print(m,'pos',np.round(r['steer'][m]['pos'],1).tolist()); print(' part',[np.round(p,2).tolist() for p in np.array(r['steer'][m]['part'])[[0,5,10]]])
