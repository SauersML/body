import numpy as np, json, glob, sys
sys.path.insert(0,'.')
import sim3 as sim
T=10
def run(kind,n,seed):
    rng=np.random.default_rng(seed); s=[]
    for _ in range(n):
        r2=np.random.default_rng(int(rng.integers(1<<30)))
        probe=sim.make_episode(np.random.default_rng(r2.integers(1<<30)),T,kind)
        env0=probe["env"]; env=sim.Env(probe["S0q"],np.random.default_rng(0),wall=env0.wall,obj=env0.obj)
        pr=np.random.default_rng(int(rng.integers(1<<30)))
        for t in range(T-1): env.step(int(pr.integers(sim.NA)))
        s.append(bool(sim.success(kind,sim.summarize(env))))
    return int(sum(s)),n
out={k:run(k,2000,11+i) for i,k in enumerate(["reach","find","wall"])}
mod={}
for f in glob.glob('$BODY_RUNS*/train_Qwen3-8B_*.json'):
    cl=json.load(open(f))['closed_loop']; a='named' if '_named_' in f else 'nameless'
    for k,(r,acts) in cl.items():
        n=round(sum(acts)/9); mod.setdefault((a,k),[0,0]); mod[(a,k)][0]+=round(r*n); mod[(a,k)][1]+=n
print(out); print(mod)
json.dump(dict(random=out,model={f"{a}|{k}":v for (a,k),v in mod.items()}),open('closed_loop_ci.json','w'))
