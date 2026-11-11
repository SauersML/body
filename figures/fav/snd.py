import numpy as np, wave, json
ev=json.load(open('events.json')); sr=48000; T=68.0; t=np.arange(int(sr*T))/sr
def note(f,amp,ph=0): return amp*(np.sin(2*np.pi*f*t+ph)+0.22*np.sin(2*np.pi*2*f*t)+0.06*np.sin(2*np.pi*3*f*t))
L=np.zeros_like(t); R=np.zeros_like(t)
# slow chord changes per scene: D, Bm, G, A-ish, D
chords=[(0,[73.42,110.0,146.83,220.0,293.66,369.99]),(ev['T2'],[61.74,92.5,123.47,185.0,246.94,293.66]),(ev['T3'],[73.42,110.0,146.83,220.0,293.66,369.99]),
        (ev['T4'],[49.0,98.0,146.83,196.0,246.94,293.66]),(ev['T5'],[73.42,110.0,146.83,220.0,277.18,369.99]),(T+5,[])]
for (a,fr),(b,_) in zip(chords[:-1],chords[1:]):
    w=np.clip((t-a+1.5)/3.0,0,1)*np.clip((b+1.5-t)/3.0,0,1)
    for k,f in enumerate(fr):
        amp=0.10/(1+0.35*k)
        L+=w*note(f*(1+0.0015*np.sin(0.21*t+k)),amp); R+=w*note(f*(1-0.0015*np.sin(0.17*t+k)),amp,0.7)
rng=np.random.default_rng(1); n=rng.standard_normal(len(t)); k=np.exp(-np.arange(2000)/300.0); k/=k.sum(); n=np.convolve(n,k,'same')*0.5
sw=0.5+0.5*np.sin(2*np.pi*t/7.0-1.5); L+=n*sw*0.4; R+=np.roll(n,700)*sw*0.4
env=np.clip(t/2.5,0,1)*np.clip((T-t)/2.5,0,1); L*=env; R*=env
bed=0.30/max(abs(L).max(),abs(R).max()); L*=bed; R*=bed
pent=[587.33,659.25,739.99,880.0,987.77,1174.66]
def hit(t0,f,amp,dec,pan=0.0):
    i0=int(t0*sr); m=int(sr*min(3.0,6/dec)); tt=np.arange(m)/sr; i1=min(len(t),i0+m); tt=tt[:i1-i0]
    s=amp*np.exp(-tt*dec)*(np.sin(2*np.pi*f*tt)+0.3*np.sin(2*np.pi*2.01*f*tt)*np.exp(-tt*dec))*np.clip(tt/0.004,0,1)
    L[i0:i1]+=s*(1-pan); R[i0:i1]+=s*(1+pan)
c=0
for kind,t0 in sorted(ev['events'],key=lambda e:e[1]):
    if kind=='chime': hit(t0,880.0,0.11,1.4); hit(t0,1318.5,0.05,1.8)
    elif kind=='tick': hit(t0,pent[c%6]*2,0.035,22,0.3*np.sin(c)); c+=1
    else: hit(t0,pent[c%6],0.05,9,0.3*np.sin(c)); c+=1
x=(np.clip(np.stack([L,R],1),-1,1)*32767).astype(np.int16)
w=wave.open('bed.wav','wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes()); w.close()
