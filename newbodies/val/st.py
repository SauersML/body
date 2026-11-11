import json,re,collections,sys
d=json.load(open(sys.argv[1]))['results']
W={'robot':{'wheel':r'\bwheels?\b','gripper':r'\bgripper','camera':r'\bcamera|\blens\b','antenna':r'\bantenna','chassis':r'\bchassis'},
   'glorp':{'wheel':r'\bglorps?\b|\bwheels?\b','gripper':r'\bvasks?\b|\bgripper','camera':r'\bmirls?\b|\bcamera','antenna':r'\btesks?\b|\bantenna','chassis':r'\bbrells?\b|\bchassis'},
   'octopus':{'arm':r'\b(arms?|tentacles?)\b','sucker':r'\bsuckers?\b','mantle':r'\bmantle\b','eye':r'\beyes?\b'}}
def first(x,ws):
    x=x.lower(); best=None
    for w,p in ws.items():
        m=re.search(p,x)
        if m and (best is None or m.start()<best[1]): best=(w,m.start())
    return best[0] if best else '-'
out={}
for b,ws in W.items():
    print('==',b)
    for dn in dict.fromkeys(r['dir'] for r in d if r['chat']==b):
        rs=[r['reply'] for r in d if r['chat']==b and r['dir']==dn]
        f=collections.Counter(first(x,ws) for x in rs); a={w:sum(bool(re.search(p,x.lower())) for x in rs) for w,p in ws.items()}
        out[f'{b}|{dn}']=dict(n=len(rs),first=dict(f),any=a)
        print(f"{dn:18s} n={len(rs)} first:",dict(f)," any:",a)
json.dump(out,open(sys.argv[1].replace('.json','_counts.json'),'w'),indent=1)
