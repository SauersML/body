import numpy as np, importlib.util
sp=importlib.util.spec_from_file_location('s','$BODY_WORK/body/hires2/stim.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
L=m.LOCS; PH=[l['phrase'].replace('{p} ','') for l in L]; SP=np.array([l['xyz'] for l in L]); SR=np.array([l['region'] for l in L])
def idx(sub, side=None):
    r=[i for i,p in enumerate(PH) if sub in p and (side is None or f' {side} ' in ' '+p+' ')]
    assert r, sub; return r
def chains(side):
    s=1 if side=='left' else -1; c=side[0].upper()
    arm=[i for i in np.where((SR=='arm'+c))[0] if 'armpit' not in PH[i]]; out={}
    xs=sorted(set(np.round(abs(SP[arm,0]),1)))
    A=[[i for i in arm if round(abs(SP[i,0]),1)==x] for x in xs]
    hand=[idx(f'of {side} wrist'), idx('centre of',side)+idx('heel of',side)]
    fing=[i for i in idx('middle finger',side) if 'skin between' not in PH[i]]
    fx=sorted(set(np.round(abs(SP[fing,0]),0)))
    F=[[i for i in fing if round(abs(SP[i,0]),0)==x] for x in fx]
    out['arm']=[idx(f'{side} collarbone')]+A+hand+F
    leg=[i for i in np.where(SR=='leg'+c)[0] if 'hip' not in PH[i] or 'thigh' in PH[i]]
    leg=[i for i in leg if 'buttock' not in PH[i]]
    ys=sorted(set(np.round(SP[leg,1],1)),reverse=True)
    out['leg']=[idx('front of '+side+' hip bone')]+[[i for i in leg if round(SP[i,1],1)==y] for y in ys]+[idx('ankle',side),idx(f'top of {side} foot'),idx(f'base of {side} big toe'),idx(f'tip of {side} big toe')]
    return out
def spine():
    names=['top of '+'head','middle of forehead','groove between','dip between','front of throat','top of breastbone','middle of breastbone','bottom of breastbone','belly, just above','NAVEL','lower belly']
    r=[]
    for n in names:
        try: r.append([i for i,p in enumerate(PH) if p=='navel'] if n=='NAVEL' else idx(n))
        except AssertionError: pass
    return r
