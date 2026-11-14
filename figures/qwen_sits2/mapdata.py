import numpy as np, json
from plane import plane, ps
out={}
for tag,li,L in [('q8b',5,29),('q32b',5,51)]:
    cl,fp,s=plane(tag,li,'mean')
    # also bootstrap SE of first-person means
    out[tag]=dict(layer=L,scale=s,clouds={p:cl[p].tolist() for p in ps},fp={k:v.mean(0).tolist() for k,v in fp.items()},
                  fp_se={k:(v.std(0)/np.sqrt(len(v))).tolist() for k,v in fp.items()})
    print(tag, {k:np.round(v,1) for k,v in out[tag]['fp'].items()}, {k:np.round(v,1) for k,v in out[tag]['fp_se'].items()})
json.dump(out,open('mapdata.json','w'))
