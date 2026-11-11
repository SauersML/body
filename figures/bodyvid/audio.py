import numpy as np, soundfile as sf, pyloudnorm as pyln, sys
from scipy.signal import fftconvolve, butter, sosfilt
SR=48000
def synth(ctrl,fps,out,seed=1):
    ctrl=np.asarray(ctrl,float); ctrl=(ctrl-ctrl.min())/(np.ptp(ctrl)+1e-9)
    T=len(ctrl)/fps; n=int(T*SR); t=np.arange(n)/SR
    m=np.interp(t,np.arange(len(ctrl))/fps,ctrl)
    k=int(SR*0.25); m=np.convolve(np.pad(m,(k,k),mode='edge'),np.hanning(2*k)/np.hanning(2*k).sum(),'same')[k:-k]  # smooth glide
    rng=np.random.default_rng(seed)
    f0=98.0*2**(m*7/12)                   # G2 rising a fifth with the morph
    phase=2*np.pi*np.cumsum(f0)/SR
    voices=[(1,0.9),(1.5,0.55),(2,0.5),(3,0.28),(4.5,0.14)]
    p=2.6-1.5*m                           # spectral tilt: brighter as the map moves
    L=np.zeros(n); R=np.zeros(n)
    for vi,(ratio,amp) in enumerate(voices):
        lfo=0.75+0.25*np.sin(2*np.pi*t/(7.3+2.1*vi)+rng.uniform(0,6))
        for det,pan in [(-0.0025,0.3),(0.0025,0.7)]:
            ph=phase*ratio*(1+det)+rng.uniform(0,6)
            for h in range(1,9):
                if ratio*h*f0.max()>9000: break
                s=np.sin(h*ph)*amp*lfo/h**p
                L+=s*(1-pan); R+=s*pan
    # air: noise band whose centre follows the morph
    for ch in (L,R):
        nz=rng.standard_normal(n)
        for lo,hi,w in [(300,900,1-m),(1200,3500,m)]:
            sos=butter(2,[lo,hi],btype='band',fs=SR,output='sos'); ch+=0.05*sosfilt(sos,nz)*w
    # reverb
    ir_n=int(3.2*SR); env=np.exp(-np.arange(ir_n)/SR*2.2)
    out_ch=[]
    for ch,sd in ((L,11),(R,12)):
        ir=np.random.default_rng(sd).standard_normal(ir_n)*env; ir/=np.sqrt((ir**2).sum())
        sos=butter(1,4000,fs=SR,output='sos'); ir=sosfilt(sos,ir)
        out_ch.append(0.55*ch+0.9*fftconvolve(ch,ir)[:n])
    y=np.stack(out_ch,1)
    fade=np.clip(np.minimum(t/2.0,(T-t)/1.5),0,1)[:,None]; y*=fade
    meter=pyln.Meter(SR); y=pyln.normalize.loudness(y,meter.integrated_loudness(y),-16.0)
    pk=np.abs(y).max()
    if pk>0.89: y=np.tanh(y/0.89*1.0)*0.89; y=pyln.normalize.loudness(y,meter.integrated_loudness(y),-16.0)
    print('LUFS',round(meter.integrated_loudness(y),2),'peak',round(np.abs(y).max(),3))
    sf.write(out,y.astype(np.float32),SR,subtype='PCM_24')
if __name__=="__main__":
    synth(np.load(sys.argv[1]),60,sys.argv[2])
