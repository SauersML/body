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
# ---- harmony (start, bass, voicing)
H=[(0.0,'D2','F#3 A3 C#4 E4'),(4.3,'B1','F#3 A3 C#4 D4'),(9.3,'G1','F#3 A3 B3 D4'),(12.6,'E2','G3 B3 D4 F#4'),
   (14.4,'G1','F#3 B3 C#4 D4'),(17.6,'A1','G3 B3 D4 E4'),(20.0,'D2','F#3 A3 C#4 E4'),(23.5,'B1','E3 A3 D4 F#4'),
   (26.0,'G1','A3 B3 D4 F#4'),(29.5,'F#1','A3 C#4 E4 F#4'),(32.0,'E2','G3 B3 D4 F#4'),(34.0,'A1','G3 B3 D4 F#4'),
   (37.0,'F#1','E3 A3 C#4 E4'),(40.0,'G1','F#3 A3 B3 D4'),(42.5,'D2','F#3 A3 C#4 E4'),
   (46.0,'B1','F#3 A3 C#4 D4'),(48.0,'G1','F#3 B3 D4'),(51.5,'D2','A3 D4 E4'),(55.5,'G1','F#3 A3 B3 C#4'),(58.5,'D2','F#3 A3 C#4 E4 A4')]
pad=np.zeros((2,N)); sub=np.zeros((2,N)); lead=np.zeros((2,N)); bell=np.zeros((2,N))
for k,(t0,b,vo) in enumerate(H):
    t1=H[k+1][0] if k+1<len(H) else T; hold=t1-t0+0.3; n=int((hold+2.0)*SR); nn=vo.split()
    e=env(n,1.4,2.0,hold)
    for j,s in enumerate(nn):
        f0=hz(s); pan=-0.6+1.2*j/max(1,len(nn)-1)
        for d in (-11,-4,3,10):   # detuned saw stack
            f=np.full(n,f0*2**(d/1200))*(1+0.0015*np.sin(2*np.pi*(0.13+0.02*d)*np.arange(n)/SR))
            add(pad,t0,saw(f)*e,pan*(0.6 if d<0 else 1.0)*np.sign(d),0.055)
    fb=hz(b); fb=fb*2 if fb<45 else fb
    es=env(n,0.9,2.0,hold); add(sub,t0,(sine(np.full(n,fb))+0.18*sine(np.full(n,2*fb)))*es,0,0.22)
    add(sub,t0,tri(np.full(n,2*fb))*es,0,0.05)
cut=curve([(0,480),(13,620),(20,650),(24.5,1400),(26.3,3000),(29,2200),(33,1100),(38,800),(44,620),(47,520),(52,600),(55.5,700),(59.2,2800),(61.5,2000),(63.3,1200)])
for c in range(2): pad[c]=vlp(pad[c],cut,res=0.6)
# ---- lead: filtered saw + triangle, low register
def tone(buf,t0,dur,s,v,pan=0.0,s2=None,bright=1500):
    n=int((dur+1.4)*SR); t=np.arange(n)/SR; f0=hz(s); f=np.full(n,f0)
    if s2: g=np.clip((t-dur*0.35)/(dur*0.5),0,1); f=f0*(hz(s2)/f0)**(0.5-0.5*np.cos(np.pi*g))
    f=f*(1+0.004*np.sin(2*np.pi*5.0*t)*np.clip((t-0.4)/0.6,0,1))
    x=0.55*saw(f)+0.6*tri(f); x=vlp(x,np.full(n,bright),res=0.2,blk=4096)
    add(buf,t0,x*env(n,0.22,1.2,dur)*v,pan,0.22)
L=[(20.6,1.4,'F#4',0.8),(22.0,1.0,'A4',0.85),(23.0,2.2,'E4',0.85),(25.2,0.5,'D4',0.75),(25.7,0.4,'E4',0.8),(26.1,2.6,'F#4',1.0),
   (28.8,1.2,'A4',0.9),(30.0,2.2,'E4',0.85),(32.3,1.5,'D4',0.8),
   (34.5,1.5,'C#4',0.8),(36.0,1.0,'D4',0.8),(37.0,2.6,'E4',0.85),(40.0,1.0,'D4',0.8),(41.0,1.4,'B3',0.75),(42.5,0.9,'A3',0.75),(43.4,2.4,'D4',0.75)]
for t0,d,s,v in L: tone(lead,t0,d,s,v)
# mirror: two voices in unison while folded, then contrary motion as the book opens
for p in (-0.75,0.75): tone(lead,51.8,3.6,'A3',0.75,p)
UP=[(55.5,1.4,'A3','B3'),(56.9,1.3,'B3','D4'),(58.2,2.6,'D4','E4')]; DN=[(55.5,1.4,'A3','G3'),(56.9,1.3,'G3','F#3'),(58.2,2.6,'F#3','D3')]
for t0,d,a,b in UP: tone(lead,t0,d,a,0.8,0.75,b,bright=1700)
for t0,d,a,b in DN: tone(lead,t0,d,a,0.8,-0.75,b,bright=1700)
# ---- soft bells where the touch dots land, and at the start
def bl(t0,s,v,pan=0.0):
    n=int(4*SR); t=np.arange(n)/SR; f=hz(s)
    x=np.sin(2*np.pi*f*t)*np.exp(-t/1.4)+0.25*np.sin(2*np.pi*2.0*f*t)*np.exp(-t/0.6)+0.08*np.sin(2*np.pi*3.01*f*t)*np.exp(-t/0.3)
    add(bell,t0,x*np.minimum(1,t/0.004)*v,pan,0.16)
for t0,s,v,p in [(0.8,'A4',0.6,-0.2),(2.0,'F#4',0.5,0.2),(6.1,'D5',0.8,0.4),(6.13,'A4',0.45,0.4),(11.1,'E5',0.8,-0.4),(11.13,'B4',0.45,-0.4),
                 (26.1,'A4',0.4,0),(59.2,'A4',0.45,0),(59.25,'E5',0.3,0.2)]: bl(t0,s,v,p)
# ---- reverb
def ir(rt=2.8):
    n=int(rt*1.2*SR); t=np.arange(n)/SR; r=np.zeros((2,n))
    for c in range(2): r[c]=rng.standard_normal(n)*np.exp(-6.9*t/rt)
    b,a=butter(2,4500/(SR/2)); r=lfilter(b,a,r,axis=1); r[:,:int(0.02*SR)]=0; return r/np.sqrt((r**2).sum(1,keepdims=True))
R=ir(); send=0.32*pad+0.45*lead+0.55*bell
wet=np.stack([fftconvolve(send[c],R[c])[:N] for c in range(2)])
mix=pad+sub+lead+bell+0.55*wet
b,a=butter(2,30/(SR/2),'high'); mix=lfilter(b,a,mix,axis=1)
b,a=butter(1,7000/(SR/2)); mix=lfilter(b,a,mix,axis=1)      # keep the top soft
t=np.arange(N)/SR; mix*=np.minimum(1,t/0.8)*np.clip((T-t)/1.6,0,1)**1.5
import pyloudnorm as pyln
meter=pyln.Meter(SR); lu=meter.integrated_loudness(mix.T); mix*=10**((-16-lu)/20)
tp=np.abs(resample_poly(mix,4,1,axis=1)).max(); print('LUFS before',round(lu,2),'TP dBTP',round(20*np.log10(tp),2))
if tp>10**(-1.2/20):  # soft knee limiter on the peaks only
    th=10**(-2.5/20); x=mix; m=np.abs(x)>th; x[m]=np.sign(x[m])*(th+(1-th)*np.tanh((np.abs(x[m])-th)/(1-th))); mix=x*10**(-1.2/20)
    lu=meter.integrated_loudness(mix.T); mix*=10**((-16-lu)/20); tp=np.abs(resample_poly(mix,4,1,axis=1)).max()
print('LUFS',round(meter.integrated_loudness(mix.T),2),'TP',round(20*np.log10(tp),2))
sf.write('m7.wav',mix.T.astype(np.float32),SR,subtype='PCM_24')
