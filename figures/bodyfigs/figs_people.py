# clean pictogram figures, local coords: centred on x=0, feet at y~640
def cap(a,b,w,c):
    return f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{c}" stroke-width="{w}" stroke-linecap="round"/>'
def _body(c, legs=True):
    s=[]
    s.append(f'<circle cx="0" cy="62" r="52" fill="{c}"/>')
    # torso: smooth shoulders to hips
    s.append(f'<path d="M-92,150 Q-100,136 -80,132 L80,132 Q100,136 92,150 L74,340 Q72,356 56,356 L-56,356 Q-72,356 -74,340 Z" fill="{c}" stroke="{c}" stroke-width="18" stroke-linejoin="round"/>')
    # arms
    for sx in (-1,1):
        s.append(cap((sx*104,160),(sx*126,300),40,c)); s.append(cap((sx*126,300),(sx*138,410),34,c))
        if legs: s.append(cap((sx*38,360),(sx*44,620),50,c))
    return s
def person(c):
    return ''.join(_body(c))
def woman(c):
    s=[]
    # long hair behind head
    s+= _body(c, legs=False)
    # dress
    s.append(f'<path d="M-66,300 L66,300 L118,490 Q0,506 -118,490 Z" fill="{c}" stroke="{c}" stroke-width="14" stroke-linejoin="round"/>')
    for sx in (-1,1): s.append(cap((sx*36,480),(sx*40,620),38,c))
    return ''.join(s)
def mannequin(c, joint='#7a3a10'):
    s=[]; wood=c
    E=lambda cx,cy,rx,ry,rot=0,f=None: f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate({rot} {cx} {cy})" fill="{f or wood}"/>'
    J=lambda x,y,r=15: f'<circle cx="{x}" cy="{y}" r="{r}" fill="{joint}"/>'
    import math
    def limb(a,b,w):
        cx=(a[0]+b[0])/2; cy=(a[1]+b[1])/2; L=math.hypot(b[0]-a[0],b[1]-a[1]); rot=math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))
        return E(cx,cy,L/2,w/2,rot)
    # stand
    s.append(f'<rect x="-6" y="380" width="12" height="250" fill="{joint}"/>')
    s.append(f'<ellipse cx="0" cy="636" rx="120" ry="16" fill="{joint}"/>')
    # head (egg) and neck
    s.append(E(0,60,40,54)); s.append(f'<rect x="-11" y="108" width="22" height="34" fill="{joint}"/>')
    # chest, waist ball, pelvis
    s.append(f'<path d="M-82,150 Q0,128 82,150 Q74,250 34,276 L-34,276 Q-74,250 -82,150 Z" fill="{wood}"/>')
    s.append(J(0,290,24))
    s.append(f'<path d="M-62,312 Q0,300 62,312 Q60,370 0,382 Q-60,370 -62,312 Z" fill="{wood}"/>')
    for sx in (-1,1):
        sh=(sx*96,162); el=(sx*120,288); wr=(sx*134,400)
        s+= [limb(sh,el,40),limb(el,wr,32),E(sx*138,430,15,24,sx*-8),J(*sh,18),J(*el,15),J(*wr,12)]
        hp=(sx*36,374); kn=(sx*44,500); an=(sx*46,604)
        s+= [limb(hp,kn,50),limb(kn,an,38),E(sx*56,620,26,12),J(*hp,18),J(*kn,16),J(*an,12)]
    return ''.join(s)
