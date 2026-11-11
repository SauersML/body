exec(open('a1.py').read().split("GN=")[0])
from collections import Counter
def nearest_region(X, ref):
    # for each spot in X, region of nearest spot in ref (excluding same spot? no, include)
    d=cdist(X,ref); return [reg[j] for j in d.argmin(1)], [ph[j] for j in d.argmin(1)]
hr=reg=='handR'
for c in ['told_N:R_hand_gone','told_A:R_hand_gone','sham:R_hand_gone','persist:R_hand_gone','told_N:L_fin','told_N:L_hand_huge']:
    X,Nw=load(c); nr,np_=nearest_region(X,N)
    side='handL' if 'L_' in c else 'handR'; m=reg==side
    print(c, Counter([nr[i] for i in np.where(m)[0]]))
    # where is self in neutral: rank of own spot
    d=cdist(X[m],N); own=np.where(m)[0]
    print('  own-spot nearest frac', np.mean([d[k].argmin()==own[k] for k in range(len(own))]))
    # direction analysis: hand centroid shift vs (forearm_lower - hand) and vs stump probe and vs neutral hand
X,Nw=load('told_N:R_hand_gone')
fa=np.array(['forearm' in p and 'right' in p for p in ph]); wr=np.array(['wrist' in p and 'right' in p for p in ph])
print('wrist spots',[p for p,w in zip(ph,wr) if w])
hand_only=hr&~wr
for nm,Y in [('neutral',N),('gone',X),('sham',load('sham:R_hand_gone')[0]),('act',load('told_A:R_hand_gone')[0]),('persist',load('persist:R_hand_gone')[0]),('persist neutral',load('persist:neutral')[0])]:
    hc=Y[hand_only].mean(0); fc=Y[fa].mean(0); lc=Y[reg=='handL'].mean(0); wc=Y[wr].mean(0)
    print(f"{nm:15s} |hand-forearm| {np.linalg.norm(hc-fc)/unit:.2f} |hand-wrist| {np.linalg.norm(hc-wc)/unit:.2f} |handR-handL| {np.linalg.norm(hc-lc)/unit:.2f} |hand-origin| {np.linalg.norm(hc)/unit:.2f}  spread {cdist(Y[hand_only],Y[hand_only]).mean()/unit:.2f}")
X,Nw=load('told_N:R_hand_gone'); st=Nw[9]
print('stump probe dist to gone-hand centroid', np.linalg.norm(X[hand_only].mean(0)-st)/unit, ' neutral hand to neutral stump', np.linalg.norm(N[hand_only].mean(0)-NwN[9])/unit)
d=np.linalg.norm(X-N,axis=1)/unit; o=np.argsort(-d)
for k in o[:25]: print(round(d[k],2),reg[k],ph[k])
print('mean by region', {g: round(float(d[reg==g].mean()),2) for g in sorted(set(reg))})
