import json,sys,numpy as np
def W(k,n,z=1.96):
    p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return p,c-h,c+h
for f in sys.argv[1:]:
    R=json.load(open(f)); print('##',f.split('/')[-2] if '/' in f else f)
    ex=R['expert']
    row=lambda name,d:print(f"{name:8s}",' '.join(f"{k.replace('poke_reach','twt')}: %.2f[%.2f,%.2f]"%W(sum(v),len(v)) for k,v in d.items()))
    row('expert',{k:ex[k]['new_expert'] for k in ex if k.startswith('new')})
    for s in R['stages']:
        if 'eval' not in s: continue
        e=s['eval']; row(s['stage'],{k:e[k]['ok'] for k in e if k.startswith('new')})
        row(' spot5',{k:list(np.array(e[k]['spot_cm'])<5) for k in e if k.startswith('new')})
