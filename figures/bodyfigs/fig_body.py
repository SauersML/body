# capsule body, column-local px. person's left = viewer's right (x>CX)
CX=380
def mir(p): return (2*CX-p[0],p[1])
# left-side skeleton
SH=(492,238); EL=(560,410); WR=(608,556); PALM=(628,606)
HIP=(440,515); KN=(458,712); AN=(466,872)
FING={'thumb':((650,590),(690,628)),'index':((640,628),(664,690)),'middle':((628,634),(638,702)),'ring':((614,632),(612,694)),'little':((602,624),(588,676))}
def lerp(a,b,t): return (a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)
PTS_L={'shoulder':(486,236),'upperarm':lerp(SH,EL,.5),'elbow':EL,'forearm':lerp(EL,WR,.5),'wrist':WR,
 'palm':(622,602),'backhand':(648,560),
 'thumb':FING['thumb'][1],'index':FING['index'][1],'middle':FING['middle'][1],'ring':FING['ring'][1],'little':FING['little'][1],
 'hip':(462,500),'thigh':lerp(HIP,KN,.5),'knee':KN,'shin':lerp(KN,AN,.5),'ankle':AN,'heel':(448,912),'sole':(478,926),'toes':(512,912)}
CEN={'crown':(CX,34),'forehead':(CX,66),'cheekL':(422,112),'cheekR':(338,112),'lips':(CX,142),'tongue':(CX,170) ,
     'earL':(458,100),'earR':(302,100),'neckF':(CX,200),'neckB':(CX+0,200),'chest':(CX,290),'upback':(CX,350),'belly':(CX,415),'lowback':(CX,470)}
def pos(key):
    if key in CEN: return CEN[key]
    side=key[-1]; base=key[:-1]
    p=PTS_L[base]; return p if side=='L' else mir(p)
def caps(a,b,w,col):
    return f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{col}" stroke-width="{w}" stroke-linecap="round"/>'
def body(kind, fill="#eceef2", ink="#3a3d45"):
    """kind: you|woman|doll"""
    segs=[]
    for m in (lambda p:p, mir):
        segs+= [(m(SH),m(EL),58),(m(EL),m(WR),50),(m(WR),m(PALM),40),(m(PALM),m((634,612)),60)]
        segs+= [(m(a),m(b),20) for a,b in FING.values()]
        segs+= [(m(HIP),m(KN),76),(m(KN),m(AN),64),(m(AN),m((470,905)),50),(m((450,910)),m((515,912)),38)]
    torso=f'M{CX-112},228 Q{CX-118},212 {CX-80},206 L{CX+80},206 Q{CX+118},212 {CX+112},228 L{CX+100},400 Q{CX+108},470 {CX+118},535 L{CX-118},535 Q{CX-108},470 {CX-100},400 Z'
    o=[]
    hair_back=''
    if kind=='woman':
        hair_back=f'<path d="M{CX-80},90 Q{CX-92},20 {CX},14 Q{CX+92},20 {CX+80},90 L{CX+92},250 Q{CX},270 {CX-92},250 Z" fill="#6b5a4e"/>'
    o.append(hair_back)
    for w_add,col in ((10,ink),(0,fill)):
        for a,b,w in segs: o.append(caps(a,b,w+w_add,col))
        o.append(f'<path d="{torso}" fill="{col}" stroke="{col}" stroke-width="{w_add}" stroke-linejoin="round"/>')
        o.append(caps((CX,170),(CX,215),54+w_add,col))
        o.append(f'<ellipse cx="{CX}" cy="96" rx="{70+w_add/2}" ry="{80+w_add/2}" fill="{col}"/>')
        o.append(f'<ellipse cx="{CX+76}" cy="100" rx="{12+w_add/2}" ry="{20+w_add/2}" fill="{col}"/><ellipse cx="{CX-76}" cy="100" rx="{12+w_add/2}" ry="{20+w_add/2}" fill="{col}"/>')
    if kind=='woman':
        o.append(f'<path d="M{CX-72},92 Q{CX-70},18 {CX},16 Q{CX+70},18 {CX+72},92 Q{CX+40},40 {CX},42 Q{CX-40},40 {CX-72},92 Z" fill="#6b5a4e"/>')
    if kind=='doll':
        # yarn hair loops, stitched seams, button eyes
        for i in range(9):
            a=-150+i*15; import math
            x=CX+74*math.cos(math.radians(a-0)); 
        for dx in range(-60,61,20):
            o.append(f'<circle cx="{CX+dx}" cy="{30+abs(dx)*0.45}" r="16" fill="none" stroke="#e8a48c" stroke-width="7"/>')
        for m in (lambda p:p, mir):
            for p in (m(SH),m(HIP)):
                o.append(f'<path d="M{p[0]-16},{p[1]-14} L{p[0]+16},{p[1]+14} M{p[0]+16},{p[1]-14} L{p[0]-16},{p[1]+14}" stroke="{ink}" stroke-width="5" opacity=".0"/>')
        o.append(f'<line x1="{CX}" y1="225" x2="{CX}" y2="530" stroke="{ink}" stroke-width="5" stroke-dasharray="12 12" opacity=".55"/>')
        for x in (CX-38,CX+38):
            o.append(f'<circle cx="{x}" cy="84" r="12" fill="{ink}"/>')
    return ''.join(o)
