# Score for body_video_combined_v3: warm voice-led maj7/add9 chords, soft saw pads whose filter opens at the
# two reveals (morph arrival ~26 s, book opening ~59 s), a gentle sub, a low lead, two mirrored voices at the end.
import numpy as np, soundfile as sf
from scipy.signal import lfilter, butter, fftconvolve, resample_poly
SR=48000; T=63.3; N=int(T*SR); rng=np.random.default_rng(7)
NM={'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
def hz(s): n,o=s[:-1],int(s[-1]); return 440*2**((NM[n]+12*(o+1)-69)/12)
def env(n,a,r,hold):  # attack, sustain for hold samples, release
    t=np.arange(n)/SR; e=np.minimum(1,t/a); e=0.5-0.5*np.cos(np.pi*e)
    rel=np.clip((t-hold)/r,0,1); return e*(0.5+0.5*np.cos(np.pi*rel))
def saw(f):  # polyBLEP saw, f: per-sample freq array
    dt=f/SR; ph=(rng.random()+np.cumsum(dt))%1; y=2*ph-1
    m=ph<dt; x=ph[m]/dt[m]; y[m]-=x+x-x*x-1
    m=ph>1-dt; x=(ph[m]-1)/dt[m]; y[m]-=x*x+x+x+1
    return y
def sine(f): return np.sin(2*np.pi*(rng.random()+np.cumsum(f/SR)))
def tri(f): ph=(rng.random()+np.cumsum(f/SR))%1; return 4*np.abs(ph-0.5)-1
def add(buf,t0,x,pan=0.0,g=1.0):
    i=int(t0*SR); x=x[:max(0,N-i)]; L=np.cos((pan+1)*np.pi/4); R=np.sin((pan+1)*np.pi/4)
    buf[0,i:i+len(x)]+=x*L*g*np.sqrt(2); buf[1,i:i+len(x)]+=x*R*g*np.sqrt(2)
def vlp(x,cut,res=0.5,blk=256):  # time-varying 4-pole lowpass (two biquads), cut: per-sample Hz
    y=x.copy()
    for _ in range(2):
        out=np.zeros_like(y); z=np.zeros(2)
        for s in range(0,len(y),blk):
            w0=2*np.pi*min(cut[s],SR*0.45)/SR; al=np.sin(w0)/(2*(0.707+res*0.3)); c=np.cos(w0)
            b=np.array([(1-c)/2,1-c,(1-c)/2]); a=np.array([1+al,-2*c,1-al]); b/=a[0]; a/=a[0]
            out[s:s+blk],z=lfilter(b,a,y[s:s+blk],zi=z)
        y=out
    return y
def curve(K):  # (time, value) keys, cosine-eased in log space
    t=np.arange(N)/SR; v=np.zeros(N)
    for (a,x),(b,y) in zip(K[:-1],K[1:]):
        m=(t>=a)&(t<=b); s=(t[m]-a)/(b-a); s=0.5-0.5*np.cos(np.pi*s); v[m]=np.exp(np.log(x)+(np.log(y)-np.log(x))*s)
    v[t<K[0][0]]=K[0][1]; v[t>K[-1][0]]=K[-1][1]; return v
