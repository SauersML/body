import json
S=json.load(open('web/stages.json')); N=len(S['Y']); PB=N-3   # first chat-tuning stage (sft)
INTRO,M,MP,BREAK,BEAT,OUTRO=2.0,0.5,0.6,1.8,4.6,1.2
h=[0.4]*N; h[0]=0.8; h[PB-1]=0.9
for i in range(PB,N): h[i]=1.0
t=INTRO; st=[]; ms=[]; mo=[]; brk=None
for i in range(N):
    st.append(t); t+=h[i]
    if i==PB-1: brk=t; t+=BREAK
    if i<N-1: ms.append(t); mo.append(MP if i+1>=PB else M); t+=mo[-1]
b1=t; b2=t+BEAT; end=b2+BEAT+OUTRO
S['sched']=dict(start=st,mstart=ms,morph=mo,PB=PB,brk=brk,BREAK=BREAK,beat1=b1,beat2=b2,BEAT=BEAT,T=end)
json.dump(S,open('web/stages.json','w')); print('break',round(brk,2),'sft',round(st[PB],2),'beats',round(b1,2),round(b2,2),'T',round(end,2))
