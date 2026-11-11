import sys,numpy as np,soundfile as sf,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt,librosa,pyloudnorm as pl
x,sr=sf.read(sys.argv[1]);m=x.mean(1)
S=librosa.amplitude_to_db(np.abs(librosa.stft(m,n_fft=4096,hop_length=512)),ref=np.max)
fig,ax=plt.subplots(2,1,figsize=(20,9),gridspec_kw={'height_ratios':[3,1]})
librosa.display.specshow(S,sr=sr,hop_length=512,y_axis='log',x_axis='s',ax=ax[0],vmin=-80);ax[0].set_ylim(30,12000)
for t in [15.93,20.93,36.93,52.43]: ax[0].axvline(t,color='w',lw=1)
meter=pl.Meter(sr,block_size=0.4)
ts=np.arange(0,len(m)/sr-0.4,0.1); st=[meter.integrated_loudness(x[int(t*sr):int((t+0.41)*sr)]) if True else 0 for t in ts]
ax[1].plot(ts,st);ax[1].set_xlim(0,len(m)/sr);ax[1].grid()
plt.tight_layout();plt.savefig(sys.argv[2],dpi=60)
print('integrated',meter.integrated_loudness(x))
for a,b in [(0,15.93),(15.93,20.93),(20.93,36.93),(36.93,52.43),(52.43,68)]: print(a,b,round(meter.integrated_loudness(x[int(a*sr):int(b*sr)]),1))
