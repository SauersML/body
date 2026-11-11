exec(open('a3.py').read().split("hr=reg==")[0])
hr=reg=='handR'; fl=reg=='footL'; wr=np.array(['wrist' in p for p in ph])&hr; hand=hr&~wr
th=np.mean([ (L[n][0][hand]-NEU[hand]).mean(0) for n in grp('gone')],0)
tf=np.mean([ (L[n][0][fl]-NEU[fl]).mean(0) for n in grp('footgone')],0)
tc=np.mean([ (L[n][0][hand]-NEU[hand]).mean(0) for n in grp('ctrl')],0)
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print('cos hand-loss shift vs foot-loss shift', cos(th,tf), ' vs ctrl shift', cos(th,tc))
# direction from neutral hand centroid to neutral footR centroid
fr=reg=='footR'
print('cos(hand shift, footR-handR)', cos(th, NEU[fr].mean(0)-NEU[hand].mean(0)), 'cos(foot shift, footR-footL)', cos(tf, NEU[fr].mean(0)-NEU[fl].mean(0)))
# project all neutral spots onto th: which spots are most along +th?
pj=NEU@th/np.linalg.norm(th); o=np.argsort(-pj); print('top along loss dir', [ph[k] for k in o[:8]]); print('bottom', [ph[k] for k in o[-5:]])
print('hand mean proj', pj[hand].mean(), 'foot', pj[fl].mean(), 'all std', pj.std())
