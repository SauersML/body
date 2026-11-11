# owner axis per-sentence scores in the earlier embodiment chat (prefix.json), split-half held out, for figure (a)
from lib7 import *; import stimuli as S, stim_own as SO
PRE=json.load(open("prefix.json")); U=lambda t:chat(PRE+[{"role":"user","content":t}]); OUT={}; L24=24
rows=[r for r in S.build() if r["cond"] in("self","other","object")]
_,X=run([U(r["text"]) for r in rows],want_states=True,layers=[L24]); X=X[:,0]
cond=np.array([r["cond"] for r in rows]); tpl=np.array([r["tpl"] for r in rows]); A=tpl%2==0
uY=X[A&(cond=="self")].mean(0)-X[A&(cond=="other")].mean(0); un=uY/np.linalg.norm(uY)
lo=X[A&(cond=="other")].mean(0)@un; hi=X[A&(cond=="self")].mean(0)@un; sc=lambda Z:((Z@un)-lo)/(hi-lo)
OUT["axis_heldout"]={c:sc(X[~A&(cond==c)]).tolist() for c in ["self","other","object"]}
OUT["axis_heldout_text"]={c:[r["text"] for r,a in zip(rows,A) if not a and r["cond"]==c] for c in ["self","other","object"]}
fac={}
for name,pr in SO.FACETS.items():
    _,Z=run([U(t) for p in pr for t in p],want_states=True,layers=[L24]); s=sc(Z[:,0])
    fac[name]=dict(you=s[0::2].tolist(),her=s[1::2].tolist(),text=[list(p) for p in pr])
OUT["facets"]=fac; OUT["n_axis_pairs"]=int((A&(cond=="self")).sum()); np.save("uY24_emb.npy",uY)
json.dump(OUT,open("res7b.json","w"))
log("axis", {c:round(float(np.mean(v)),2) for c,v in OUT["axis_heldout"].items()}, {k:(round(np.mean(v["you"]),2),round(np.mean(v["her"]),2)) for k,v in fac.items()})
