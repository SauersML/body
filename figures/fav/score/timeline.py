import json, os
HERE=os.path.dirname(os.path.abspath(__file__)); FAV=os.path.dirname(HERE)
D=json.load(open(f'{FAV}/fa_nn.json'))['left']
SIDE=D['side']; POS=D['pos']
PHR=[p.replace('{p}','your') for p in D['phr']]
def idx(s): return PHR.index(s)
def load():
    d=json.load(open(f'{FAV}/timing.json'))
    sc=d['scenes']; T=dict(T2=sc[1]['start'],T3=sc[2]['start'],T4=sc[3]['start'],T5=sc[4]['start'],TEND=sc[4]['end'])
    ev=[]
    for k,s in enumerate(d['spot_appear']):
        i=idx(s['spot']); ev.append(dict(kind='spot',t=s['t'],i=i,side=SIDE[i],pos=POS[i],example=(k==0)))
    for L in d['links']:
        i,j=idx(L['from']),idx(L['to'])
        ev.append(dict(kind='linkA' if L['model']=='Qwen3-32B' else 'linkB',t=L['start'],end=L['end'],i=i,j=j,example=L['example']))
    return T,ev,float(d['video_length_s']),d['captions']
if __name__=='__main__':
    T,ev,dur,cap=load(); print(T,dur,len(ev)); from collections import Counter; print(Counter(e['kind'] for e in ev))
    print(sorted(set(POS)))
