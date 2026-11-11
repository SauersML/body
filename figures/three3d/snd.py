import numpy as np, wave
sr=48000; T=13.0; t=np.arange(int(sr*T))/sr
def note(f, amp, ph=0): 
    return amp*(np.sin(2*np.pi*f*t+ph)+0.25*np.sin(2*np.pi*2*f*t)+0.08*np.sin(2*np.pi*3*f*t))
# soft pad: D major-ish open voicing, slow chorus detune
fr=[73.42,110.0,146.83,220.0,293.66,369.99]
L=np.zeros_like(t); R=np.zeros_like(t)
for k,f in enumerate(fr):
    a=0.10/(1+0.35*k)
    L+=note(f*(1+0.0015*np.sin(0.21*t+k)),a); R+=note(f*(1-0.0015*np.sin(0.17*t+k)),a,0.7)
# gentle breath noise, lowpassed
rng=np.random.default_rng(1); n=rng.standard_normal(len(t))
k=np.exp(-np.arange(2000)/300.0); k/=k.sum(); n=np.convolve(n,k,'same')*0.6
sw=0.5+0.5*np.sin(2*np.pi*t/6.5-1.5)
L+=n*sw*0.5; R+=np.roll(n,700)*sw*0.5
# soft bell at rotation start and end
for t0,f in [(1.2,587.33),(12.0,880.0)]:
    e=np.where(t>t0,np.exp(-(t-t0)*1.6),0)
    b=0.06*e*np.sin(2*np.pi*f*(t-t0))*(t>t0); L+=b; R+=b
env=np.clip(t/2.0,0,1)*np.clip((T-t)/2.0,0,1); env=env**1.5
L*=env; R*=env; m=max(abs(L).max(),abs(R).max()); L*=0.35/m; R*=0.35/m
x=(np.stack([L,R],1)*32767).astype(np.int16)
w=wave.open('/tmp/fa_bed.wav','wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes()); w.close()
