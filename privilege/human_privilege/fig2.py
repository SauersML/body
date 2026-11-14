import sys,os,json,numpy as np
D=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,D+'/hpriv'); from bodies import xyz
Z=np.load(f'{D}/hpriv/out/q32_states.npz',allow_pickle=True); meta=Z['meta']
bo=np.array([m[0] for m in meta]); pi=np.array([int(m[1]) for m in meta])
X=Z['L31'].astype(np.float32); m=bo=='human'
P=np.stack([X[m][pi[m]==i].mean(0) for i in range(16)]); P-=P.mean(0); P/=np.linalg.norm(P,axis=1,keepdims=True)
Mm=1-P@P.T; c=xyz('human'); Mr=np.linalg.norm(c[:,None]-c[None],axis=-1); order=np.argsort(-c[:,2]+0.001*c[:,0])
PUR='#7B3FE4'
def heat(M,x0,y0,s):
    M=M[np.ix_(order,order)]; iu=~np.eye(16,dtype=bool); lo,hi=M[iu].min(),M[iu].max(); o=[]
    for i in range(16):
        for j in range(16):
            a=1 if i==j else 1-(M[i,j]-lo)/(hi-lo); o.append(f'<rect x="{x0+j*s}" y="{y0+i*s}" width="{s+.5}" height="{s+.5}" fill="#1D1D1F" fill-opacity="{a:.3f}"/>')
    return ''.join(o)
R={'q32':json.load(open(f'{D}/res_q32.json'))['rsa']['31'],'q8':json.load(open(f'{D}/res_q8.json'))['rsa']['19']}
N=json.load(open(f'{D}/null_rsa.json'))
rows=['human','dog','robot','bird','octopus','velk','GAP','snake','tree']
lab={'human':'human','dog':'dog','robot':'robot','bird':'bird','octopus':'octopus','velk':'made-up body','snake':'snake','tree':'tree'}
COL={'human':'#D2432F','velk':'#F2A497'}; LCOL={'human':'#D2432F','velk':'#D2432F'}
W=2400;H=2120; s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica Neue,Arial" font-size="48" fill="#1D1D1F"><rect width="{W}" height="{H}" fill="#fff"/>']
s.append('<text x="60" y="90" font-size="58" font-weight="700">Same human layout, made-up part names: the body map falls to chance</text>')
# method
s.append('<text x="60" y="200" font-weight="700">"Something lightly touches your left elbow."</text><text x="60" y="265">16 body parts × 8 sentences</text>')
s.append('<text x="60" y="360" font-size="70">↓</text>')
sz=17; hx=170; hy=310; s.append(heat(Mm,hx,hy,sz)); s.append(f'<text x="{hx+16*sz+30}" y="{hy+120}">distances between</text><text x="{hx+16*sz+30}" y="{hy+180}">part states inside</text><text x="{hx+16*sz+30}" y="{hy+240}">the model</text>')
hx2=1160; s.append(f'<text x="1000" y="{hy+160}" font-size="64" font-weight="700">vs</text>'); s.append(heat(Mr,hx2,hy,sz)); s.append(f'<text x="{hx2+16*sz+30}" y="{hy+120}">real distances</text><text x="{hx2+16*sz+30}" y="{hy+180}">on the body</text>')
s.append(f'<text x="1900" y="{hy+120}" font-size="64">→</text><text x="1990" y="{hy+120}" font-weight="700">match</text><text x="1990" y="{hy+180}">(rank</text><text x="1990" y="{hy+240}">correlation)</text>')
# panels
y0=800; rh=118; bh=74; sc=1000/0.62; xs={'q32':560,'q8':1540}; sc=700/0.62
for t,nm in [('q32','Qwen3-32B'),('q8','Qwen3-8B')]:
    x0=xs[t]; s.append(f'<text x="{x0}" y="{y0-40}" font-weight="700">{nm}</text>')
    y=y0
    for b in rows:
        if b=='GAP': y+=60; continue
        v,lo,hi=R[t][b]; nmed,n95,_=N[t][b]
        s.append(f'<rect x="{x0+min(nmed,0)*sc}" y="{y}" width="{(n95-min(nmed,0))*sc}" height="{rh-14}" fill="#E4E4EA"/>')
        col=COL.get(b,PUR)
        if v>0: s.append(f'<rect x="{x0}" y="{y+(rh-14-bh)/2}" width="{v*sc}" height="{bh}" rx="6" fill="{col}"/>')
        s.append(f'<line x1="{x0+max(lo,0)*sc}" y1="{y+(rh-14)/2}" x2="{x0+hi*sc}" y2="{y+(rh-14)/2}" stroke="#1D1D1F" stroke-width="5"/>')
        s.append(f'<text x="{x0+max(hi,0.03)*sc+18}" y="{y+(rh-14)/2+17}" font-weight="700">{v:.2f}</text>')
        if t=='q32': s.append(f'<text x="80" y="{y+(rh-14)/2+17}" fill="{LCOL.get(b,"#1D1D1F")}" font-weight="{700 if b in COL else 400}">{lab[b]}</text>')
        y+=rh
    s.append(f'<line x1="{x0}" y1="{y0-10}" x2="{x0}" y2="{y}" stroke="#1D1D1F" stroke-width="4"/>')
    s.append(f'<text x="{x0}" y="{y+60}">0</text><text x="{x0+0.5*sc}" y="{y+60}" text-anchor="middle">0.5</text>')
yv=y0+5*rh
s.append(f'<text x="80" y="{yv+(rh-14)/2+70}" font-size="44" fill="#D2432F">human layout,</text>')
s.append(f'<text x="80" y="{yv+(rh-14)/2+120}" font-size="44" fill="#D2432F">made-up names</text>')
yb=y0+9*rh+60
s.append(f'<rect x="560" y="{yb-10}" width="70" height="50" fill="#E4E4EA"/><text x="650" y="{yb+32}">chance: part names shuffled</text>')
s.append(f'<text x="1650" y="{yb+132}">bars: how well the map matches the body</text>'.replace('bars: how well the map matches the body',''))
s.append('</svg>'); open(os.path.expanduser('~/fig/human_privilege_v1.svg'),'w').write(''.join(s))
