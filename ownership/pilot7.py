from lib7 import *; import rh7stim as R
I0=R.INTRO["rubber"]; I1=I0.replace(", where your own hand would naturally be,",",")
for nm,I in [("leading",I0),("neutral",I1)]:
  R.INTRO["rubber"]=I
  for wd in R.W:
    row=[]
    for c,n in [("sync",0),("sync",2),("lag",2),("sync",24),("lag",24),("mismatch",24),("random",24)]:
      P=[chat([{"role":"user","content":R.prompt(R.events(c,np.random.default_rng(s)),"rubber",wd,n)}]) for s in range(16)]
      lo,_=run(P); row.append(f"{c}{n}:{lo.mean():.1f}±{lo.std()/4:.1f}")
    log(nm,wd," ".join(row))
R.INTRO["rubber"]=I0
P=[chat([{"role":"user","content":R.prompt(R.events(c,np.random.default_rng(s)),"rubber","paren",n,q=R.FQ["rubber"])}]) for c,n in [("sync",0),("sync",24),("lag",24)] for s in range(3)]
for g in gen(P): print("|",g.replace("\n"," ")[:220])
