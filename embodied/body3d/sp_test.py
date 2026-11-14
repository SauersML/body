import sys, numpy as np, torch, types
sys.argv=["x","Qwen/Qwen3-0.6B","named","sim"]
import train3d as TR
class FakeTok:
    pad_token_id=0
class FakeB:
    def __init__(s,*a,**k):
        s.tk=types.SimpleNamespace(prompt=lambda e,qa,upto_t=None,answer=True,goal=True: ([1]*(4*((upto_t if upto_t is not None else 20)+1)), [-100]*(4*((upto_t if upto_t is not None else 20)+1)), [0]*(4*((upto_t if upto_t is not None else 20)+1)), list(range(3,4*((upto_t if upto_t is not None else 20)+1),4)), list(range(0,4*((upto_t if upto_t is not None else 20)+1),4)), []))
        s.tok=FakeTok(); s.m=types.SimpleNamespace(eval=lambda:None)
        s.ad=types.SimpleNamespace(cap=torch.tensor(1.)); 
    def load_small(s,ck): pass
    def run(s,ids,kind,att,X): return types.SimpleNamespace(last_hidden_state=torch.zeros(ids.shape[0],ids.shape[1],4))
    def act(s,h): return torch.zeros(h.shape[0],41)
    def cand_logp(s,eps,qas): return [np.random.randn(2) for _ in eps]
TR.Body=FakeB; torch.load=lambda *a,**k: {"rn":1.0}
src=open("selfplay.py").read().split("VIDS = {}")[0]
exec(compile(src,"sp","exec"))
import time; t=time.time()
for kind in TASKS:
    eps=rollout(kind,[(4242,i) for i in range(6)],[0.0,0.1,0.2,0.3,0,0],record=2)
    print(kind,[e["info"]["success"] for e in eps],[e["info"]["expert_success"] for e in eps],[round(e["info"]["tip_cm"]) for e in eps], eps[0]["A"].shape, len(eps[0]["rec"]), round(time.time()-t,1))
print(judge(eps))
TR.Tok=lambda m,n: types.SimpleNamespace(tok=FakeTok(), prompt=lambda e,qa,goal=True: B.tk.prompt(e,qa))
for e in eps: e.pop("rec",None)
mx=Mix(eps,8,1); b=next(iter(mx)); print([x.shape if hasattr(x,"shape") else len(x) for x in b])
