import sys,os,json,numpy as np
D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,D+'/hpriv'); from bodies import xyz,names
Z=np.load(f'{D}/hpriv/out/q32_states.npz',allow_pickle=True); meta=Z['meta']
bo=np.array([m[0] for m in meta]); pi=np.array([int(m[1]) for m in meta])
X=Z['L31'].astype(np.float32); m=bo=='human'
P=np.stack([X[m][pi[m]==i].mean(0) for i in range(16)]); P-=P.mean(0); P/=np.linalg.norm(P,axis=1,keepdims=True)
Mm=1-P@P.T; c=xyz('human'); Mr=np.linalg.norm(c[:,None]-c[None],axis=-1)
order=np.argsort(-c[:,2]+0.001*c[:,0])  # head to feet
def heat(M,x0,y0,s,col):
    M=M[np.ix_(order,order)]; iu=~np.eye(16,dtype=bool); lo,hi=M[iu].min(),M[iu].max(); out=[]
    for i in range(16):
        for j in range(16):
            v=0 if i==j else (M[i,j]-lo)/(hi-lo); a=1-v
            out.append(f'<rect x="{x0+j*s}" y="{y0+i*s}" width="{s}" height="{s}" fill="{col}" fill-opacity="{a:.3f}"/>')
    return ''.join(out)
A=json.load(open(f'{D}/res_q32.json'))['rsa']['31']; pv=json.load(open(f'{D}/pval_q32_L31.json'))
lab={'human':'human','snake':'snake','tree':'tree','dog':'dog','robot':'robot','bird':'bird','octopus':'octopus','velk':'made-up body: human layout, made-up part names'}
bods=sorted(lab,key=lambda b:-A[b][0])
W=2400;H=2130
s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica Neue,Arial" font-size="50" fill="#1D1D1F"><rect width="{W}" height="{H}" fill="#fff"/>']
s.append('<text x="60" y="95" font-size="62" font-weight="700">Qwen3-32B maps the human body more truly than other limbed bodies</text>')
# method row
s.append('<rect x="60" y="150" width="760" height="400" rx="30" fill="#F4F4F6"/>')
for k,t in enumerate(['"Something lightly','touches your','left elbow."','× 16 body parts','× 8 sentences']):
    s.append(f'<text x="100" y="{235+k*72}" font-weight="{700 if k<3 else 400}">{t}</text>')
s.append('<text x="840" y="370" font-size="70">→</text>')
sz=19; s.append(heat(Mm,940,170,sz,'#7B3FE4')); s.append(f'<text x="940" y="{170+16*sz+60}">distances between</text><text x="940" y="{170+16*sz+118}">part states, layer 31</text>')
s.append(f'<text x="1300" y="370" font-size="70">vs</text>')
s.append(heat(Mr,1420,170,sz,'#1A9E77')); s.append(f'<text x="1420" y="{170+16*sz+60}">real distances</text><text x="1420" y="{170+16*sz+118}">on the body</text>')
s.append('<text x="1780" y="370" font-size="70">→</text><text x="1870" y="330" font-weight="700">match</text><text x="1870" y="395">(rank</text><text x="1870" y="455">correlation)</text>')
# bars
y0=730; bh=120; gap=36; x0=760; scale=1500/0.62
s.append(f'<line x1="{x0}" y1="{y0-30}" x2="{x0}" y2="{y0+8*(bh+gap)}" stroke="#1D1D1F" stroke-width="4"/>')
ys={}
for k,b in enumerate(bods):
    y=y0+k*(bh+gap); v=A[b][0]; ys[b]=y; col='#D2432F' if b=='human' else ('#E8590C' if b=='velk' else '#7B3FE4')
    w=max(v,0)*scale
    s.append(f'<rect x="{x0}" y="{y}" width="{w}" height="{bh}" rx="8" fill="{col}"/>')
    # CI whisker
    s.append(f'<line x1="{x0+max(A[b][1],0)*scale}" y1="{y+bh/2}" x2="{x0+A[b][2]*scale}" y2="{y+bh/2}" stroke="#1D1D1F" stroke-width="5"/>')
    if b=='velk':
        s.append(f'<text x="{x0-30}" y="{y+52}" text-anchor="end" fill="{col}" font-weight="700">made-up body</text><text x="{x0+A[b][2]*scale+40}" y="{y+50}" fill="{col}">same layout as human,</text><text x="{x0+A[b][2]*scale+40}" y="{y+108}" fill="{col}">made-up part names</text>')
    else:
        s.append(f'<text x="{x0-30}" y="{y+bh/2+18}" text-anchor="end" fill="{col}" font-weight="{700 if b=="human" else 400}">{lab[b]}</text>')
    s.append(f'<text x="{x0+A[b][2]*scale+20 if b!="velk" else x0+w+600}" y="{y+bh/2+18}" font-weight="700">{"" if b=="velk" else f"{v:.2f}"}</text>')
# velk value
yv=ys['velk']; 
s.append(f'<text x="{x0-30}" y="{yv+110}" text-anchor="end">{A["velk"][0]:.2f}</text>')
# bracket human vs velk on the right
yh=ys['human']+bh/2; bx=x0+0.62*scale+20
s.append(f'<path d="M{bx-40},{yh} H{bx} V{yv+bh/2} H{bx-40}" fill="none" stroke="#1D1D1F" stroke-width="5"/>')
s.append(f'<text x="{bx-20}" y="{(yh+yv+bh/2)/2}" text-anchor="end" transform="rotate(-90 {bx+60} {(yh+yv+bh/2)/2})">p &lt; 0.001</text>')
s.append(f'<text x="{x0}" y="{y0+8*(bh+gap)+60}">0</text><text x="{x0+0.5*scale}" y="{y0+8*(bh+gap)+60}" text-anchor="middle">0.5</text>')
s.append(f'<text x="{x0+0.25*scale}" y="{y0+8*(bh+gap)+130}" text-anchor="middle">how well the inner map matches the real body</text>')
s.append('</svg>'); open(os.path.expanduser('~/fig/human_privilege_v1.svg'),'w').write(''.join(s))
