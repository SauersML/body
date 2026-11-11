exec(open('a3.py').read().split("hr=reg==")[0])
from collections import Counter
hr=reg=='handR'; fl=reg=='footL'; wr=np.array(['wrist' in p for p in ph])&hr; hand=hr&~wr
GN=sorted(set(reg))
def cen(X,m): return X[m].mean(0)
for c,mask,side in [('gone',hand,'handR'),('ctrl',hand,'handR'),('footgone',fl,'footL'),('footctrl',fl,'footL')]:
    rows=[]
    for n in grp(c):
        X,Nw,Ex=L[n]
        # nearest region (neutral map) of moved spots
        nr=Counter(reg[cdist(X[mask],NEU).argmin(1)])
        # within own context: which region centroid nearest to the part's centroid (excluding itself)
        cc={g:cen(X,(reg==g)&~mask) for g in GN if ((reg==g)&~mask).sum()}
        hc=cen(X,mask); dd={g:np.linalg.norm(hc-v)/unit for g,v in cc.items()}
        air=np.linalg.norm(hc-Ex[:3].mean(0))/unit; air0=np.linalg.norm(cen(NEU,mask)-np.mean([L[m][2][:3].mean(0) for m in grp('neutral')],0))/unit
        # spread within part
        sp_=cdist(X[mask],X[mask]).mean()/unit
        # projection of shift on directions
        shift=cen(X,mask)-cen(NEU,mask)
        rows.append((dict(nr), sorted(dd,key=dd.get)[:3], round(air,2), round(air0,2), round(sp_,2), round(np.linalg.norm(shift)/unit,2)))
    print('==',c); [print(r) for r in rows]
print('neutral spread hand', cdist(NEU[hand],NEU[hand]).mean()/unit, 'foot', cdist(NEU[fl],NEU[fl]).mean()/unit)
# cos of shift between gone chats (consistent direction?)
sh=[ (L[n][0][hand]-NEU[hand]).ravel() for n in grp('gone')]
cs=[[float(a@b/np.linalg.norm(a)/np.linalg.norm(b)) for b in sh] for a in sh]; print(np.round(cs,2))
# is the shift a common translation (all hand spots move same direction)?
for n in grp('gone')[:3]:
    Dv=L[n][0][hand]-NEU[hand]; m=Dv.mean(0); print('common-translation share', np.linalg.norm(m)**2*hand.sum()/ (Dv**2).sum())
