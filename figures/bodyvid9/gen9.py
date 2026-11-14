# v5 (bodyvid9): gen8's score retimed through warp.py (v4 scene time -> v5 time).
# Score for body_video_combined_v3: warm voice-led maj7/add9 chords, soft saw pads whose filter opens at the
# two reveals (morph arrival ~26 s, book opening ~59 s), a gentle sub, a low lead, two mirrored voices at the end.
import numpy as np, soundfile as sf
from scipy.signal import lfilter, butter, fftconvolve, resample_poly
import sys; sys.path.insert(0,'.'); from warp import g as _g
WT=lambda x: float(_g(x))
SR=48000; T=47.8; N=int(T*SR); rng=np.random.default_rng(7)
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
    K=[(WT(a),x) for a,x in K]
    t=np.arange(N)/SR; v=np.zeros(N)
    for (a,x),(b,y) in zip(K[:-1],K[1:]):
        m=(t>=a)&(t<=b); s=(t[m]-a)/(b-a); s=0.5-0.5*np.cos(np.pi*s); v[m]=np.exp(np.log(x)+(np.log(y)-np.log(x))*s)
    v[t<K[0][0]]=K[0][1]; v[t>K[-1][0]]=K[-1][1]; return v
# ---- harmony: (start, bass, voicing). D major -> F major at the morph arrival (26 s) -> home to D (40 s)
H=[(0.0,'D2','F#3 A3 C#4 E4'),          # Dmaj9
   (4.3,'B1','F#3 A3 D4 E4'),           # Bm11
   (8.0,'G1','F#3 A3 B3 D4'),           # Gmaj9-ish (G6/9 maj7)
   (11.1,'E2','F#3 G3 B3 D4'),          # Em9
   (12.8,'A1','E3 G3 B3 D4'),           # A9sus
   (14.4,'F#1','E3 A3 C#4 E4'),         # F#m7 (halves part: opens)
   (17.2,'G1','D3 F#3 B3 C#4 E4'),      # Gmaj13#11
   (20.0,'E2','D3 G3 A3 B3 F#4'),       # Em11 (morph starts)
   (22.4,'A#1','D3 F3 A3 C4 F4'),       # Bbmaj9 (borrowed bVI)
   (24.4,'C2','A#2 D3 F3 G3 C4'),       # C9sus4 (V of F)
   (26.0,'F1','C3 G3 A3 E4 G4 C5'),     # Fmaj9 WIDE: new key, the reveal
   (29.2,'D2','C3 F3 A3 E4 G4'),        # Dm11
   (31.6,'A#1','D3 A3 C4 F4 A4'),       # Bbmaj9
   (34.0,'G1','A#2 F3 A3 D4 F4'),       # Gm9 (pivot: ii of F, iv of D) morph back
   (36.4,'A#1','D3 F3 A3 E4'),          # Bbmaj7#11
   (38.4,'A1','D3 G3 B3 E4'),           # A9sus
   (39.4,'A1','C#3 G3 B3 E4'),          # A13 resolve
   (40.0,'D2','F#3 A3 C#4 E4'),         # Dmaj9 home in the body
   (41.6,'B1','F#3 A3 D4 E4'),          # Bm11 halves close
   (43.2,'G1','F#3 A3 B3 D4'),          # Gmaj9
   (44.6,'A1','E3 G3 B3 D4'),           # A9sus
   (46.0,'D2','E3 A3 C#4 F#4'),         # Dmaj9 scene 2 (dip)
   (48.0,'G1','F#3 A3 B3 D4'),          # Gmaj9 fold begins: closing
   (50.3,'E2','G3 A3 B3 D4'),           # Em11 tighter
   (52.4,'D2','A3 D4 E4'),              # Dsus2 folded: tight, near unison
   (54.0,'D2','A#3 D4 E4'),             # Gm6/D (borrowed iv) over pedal
   (55.5,'A#1','D3 F3 A3 C4 F4'),       # Bbmaj9 opening (bloom)
   (57.4,'A1','D3 G3 B3 E4 A4'),        # A9sus
   (58.4,'A1','C#3 G3 B3 E4 A4'),       # A13
   (59.2,'D1','A2 F#3 C#4 E4 A4')]      # Dmaj9 wide: home
H=[(WT(a),b,c) for a,b,c in H]
t=np.arange(N)/SR
width=curve([(0,0.45),(14,0.5),(17.6,0.85),(26,1.0),(33,0.9),(40,0.6),(43.8,0.45),(46,0.6),(50,0.35),(52.4,0.05),(55.3,0.05),(59.2,1.0),(63.3,0.9)])
pad=np.zeros((2,N)); pad2=np.zeros((2,N)); sub=np.zeros((2,N)); bell=np.zeros((2,N))
for k,(t0,b,vo) in enumerate(H):
    t1=H[k+1][0] if k+1<len(H) else T; hold=t1-t0+0.25; n=int((hold+2.2)*SR); nn=vo.split()
    att=0.5 if t1-t0<1.5 else 1.2; e=env(n,att,2.0,hold); wd=width[int(t0*SR)]
    for j,s in enumerate(nn):
        f0=hz(s); pan=wd*(-0.8+1.6*j/max(1,len(nn)-1)); tt=np.arange(n)/SR
        for d in (-9,-3,4,10):
            f=np.full(n,f0*2**(d/1200))*(1+0.0015*np.sin(2*np.pi*(0.13+0.02*d)*tt))
            add(pad,t0,saw(f)*e,np.clip(pan+0.25*wd*np.sign(d),-1,1),0.05)
        for d in (-6,6):  # second layer: softer, one octave-shaded triangle+saw blend, panned mirrored
            f=np.full(n,f0*2**(d/1200)); add(pad2,t0,(0.6*saw(f)+0.6*tri(f))*e,-pan,0.07)
    fb=hz(b); fb=fb*2 if fb<44 else fb
    es=env(n,0.7,2.0,hold); add(sub,t0,(sine(np.full(n,fb))+0.15*sine(np.full(n,2*fb)))*es,0,0.2)
cut=curve([(0,420),(4,520),(12,600),(14.4,700),(18,950),(22,1200),(25,1700),(26.2,2900),(28,2300),(31.6,1900),(34,1300),(38,900),(40,1000),(43,700),(46,520),(48,600),(52.4,420),(55.3,480),(57.5,1300),(59.4,2700),(61.5,1700),(63.3,900)])
cut2=curve([(0,300),(14.4,500),(20,800),(26,1700),(34,1000),(40,700),(46,500),(52,380),(55.5,500),(59.4,1800),(63.3,800)])
for c in range(2): pad[c]=vlp(pad[c],cut,res=0.6); pad2[c]=vlp(pad2[c],cut2,res=0.3)
# tremolo pulse on layer 2: off at first, slow breathing while parting, quarter-ish pulse in the new key, stills at the fold
rate=curve([(0,0.8),(14.4,1.2),(20,2.2),(26,2.8),(34,2.2),(40,1.6),(46,1.2),(52.4,0.6),(55.5,1.4),(59.2,2.4),(63.3,1.6)])
depth=np.interp(t,[WT(x) for x in [0,13,15,20,26,33,40,44,46,50,52.4,55.5,58,62,63.3]],[0,0,0.35,0.5,0.7,0.6,0.4,0.3,0.25,0.3,0.05,0.2,0.5,0.4,0.3])
trem=1-depth*(0.5-0.5*np.cos(2*np.pi*np.cumsum(rate)/SR))
g2=np.interp(t,[WT(x) for x in [0,13,16,26,34,40,46,52.4,55.5,59.2,63.3]],[0,0,0.8,1.1,0.9,0.7,0.5,0.35,0.6,1.1,0.9])
pad2*=trem*g2
# dynamics: swells and dips following the scenes
dyn=curve([(0,0.55),(4,0.7),(12,0.75),(14.4,0.8),(17.6,0.9),(22,1.0),(25.6,1.15),(26.3,1.35),(29,1.15),(33,1.05),(36,0.95),(40,1.0),(43.8,0.85),(45.8,0.6),(46.3,0.5),(48,0.65),(52.4,0.5),(55.3,0.55),(58,1.0),(59.4,1.35),(61.5,1.0),(63.3,0.6)])
subg=curve([(0,0.7),(14.4,0.9),(26,1.1),(40,1.0),(46,0.7),(52.4,0.8),(55.5,0.6),(59.2,1.1),(63.3,0.8)])
pad*=dyn; pad2*=dyn; sub*=dyn*subg*(1-0.15*depth*(0.5-0.5*np.cos(2*np.pi*np.cumsum(rate)/SR)))
def bl(t0,s,v,pan=0.0):
    n=int(4*SR); tt=np.arange(n)/SR; f=hz(s)
    x=np.sin(2*np.pi*f*tt)*np.exp(-tt/1.6)+0.2*np.sin(2*np.pi*2.0*f*tt)*np.exp(-tt/0.5)
    add(bell,t0,x*np.minimum(1,tt/0.006)*v,pan,0.13)
for t0,s,v,p in [(6.1,'A4',0.6,0.3),(11.1,'B4',0.6,-0.3),(26.05,'A4',0.4,0),(59.25,'A4',0.45,0)]: bl(WT(t0),s,v,p)
def ir(rt=3.0):
    n=int(rt*1.2*SR); tt=np.arange(n)/SR; r=np.zeros((2,n))
    for c in range(2): r[c]=rng.standard_normal(n)*np.exp(-6.9*tt/rt)
    b,a=butter(2,4000/(SR/2)); r=lfilter(b,a,r,axis=1); r[:,:int(0.02*SR)]=0; return r/np.sqrt((r**2).sum(1,keepdims=True))
R=ir(); send=0.35*pad+0.35*pad2+0.5*bell
wet=np.stack([fftconvolve(send[c],R[c])[:N] for c in range(2)])
mix=pad+pad2+sub+bell+0.55*wet
b,a=butter(2,30/(SR/2),'high'); mix=lfilter(b,a,mix,axis=1)
b,a=butter(1,6000/(SR/2)); mix=lfilter(b,a,mix,axis=1)
mix*=np.minimum(1,t/0.8)*np.clip((T-t)/1.8,0,1)**1.5
import pyloudnorm as pyln
meter=pyln.Meter(SR); lu=meter.integrated_loudness(mix.T); mix*=10**((-16-lu)/20)
tp=np.abs(resample_poly(mix,4,1,axis=1)).max(); print('LUFS0',round(lu,2),'TP',round(20*np.log10(tp),2))
for _ in range(3):
    if tp<=10**(-1.1/20): break
    th=10**(-3/20); x=mix; m=np.abs(x)>th; x[m]=np.sign(x[m])*(th+(1-th)*np.tanh((np.abs(x[m])-th)/(1-th))); mix=x*10**(-1.3/20)/max(1,np.abs(x).max())
    lu=meter.integrated_loudness(mix.T); mix*=10**((-16-lu)/20); tp=np.abs(resample_poly(mix,4,1,axis=1)).max()
print('LUFS',round(meter.integrated_loudness(mix.T),2),'TP',round(20*np.log10(tp),2))
sf.write('m9.wav',mix.T.astype(np.float32),SR,subtype='PCM_24')
