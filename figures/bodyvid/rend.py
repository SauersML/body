import numpy as np, skia, subprocess, os
W,H=1920,1080; FPS=60
BASE={'head':'#7B5EA7','torso':'#5E646C','arm':'#E8833A','hand':'#D13C6A','leg':'#2F80C9','foot':'#3A9D5D'}
def hx(h): return np.array([int(h[i:i+2],16) for i in (1,3,5)])/255.
def colours(cat,side):
    out=[]
    for c,s in zip(cat,side):
        b=hx(BASE[c])
        out.append(b*0.62+0.38 if s>0 else (b*0.82 if s<0 else b*0.9+0.05))
    return np.array(out)
def smooth(x): x=np.clip(x,0,1); return x*x*x*(x*(6*x-15)+10)
def rot(yaw,pitch):
    cy,sy,cp,sp=np.cos(yaw),np.sin(yaw),np.cos(pitch),np.sin(pitch)
    Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]); Rx=np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    return Rx@Ry
def project(P,center,yaw,pitch,scale=4.6,D=900,off=(0,40)):
    Q=(P-center)@rot(yaw,pitch).T          # camera looks along -z; +z toward viewer
    f=D/(D-Q[:,2]); x=W/2+off[0]+Q[:,0]*scale*f; y=H/2+off[1]-Q[:,1]*scale*f
    return x,y,Q[:,2],f
FONT=skia.Typeface('Helvetica Neue',skia.FontStyle.Bold())
def col4(c,a): return skia.Color4f(float(c[0]),float(c[1]),float(c[2]),float(np.clip(a,0,1)))
def draw_frame(x,y,z,f,cols,rad,alpha,glow,blur=None,edges=None,ealpha=0.,texts=(),shadow=None,extra=None):
    surf=skia.Surface(W,H); cv=surf.getCanvas(); cv.clear(skia.ColorWHITE)
    fog=np.clip((z-z.min())/(np.ptp(z)+1e-6),0,1)*0.35+0.65  # 0.65 far .. 1 near
    if shadow is not None:  # soft floor shadow (x,y arrays)
        p=skia.Paint(AntiAlias=True,MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle,18))
        p.setColor4f(skia.Color4f(0.2,0.22,0.28,0.022))
        for sx,sy in zip(*shadow): cv.drawCircle(float(sx),float(sy),14,p)
    if extra: extra(cv)
    order=np.argsort(z)
    # glow
    for blur_px,scale_r,ga in [(26,3.2,glow*0.22),(9,1.7,glow*0.35)]:
        if ga<=0.003: continue
        p=skia.Paint(AntiAlias=True,MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle,blur_px))
        for i in order:
            p.setColor4f(col4(cols[i],ga*alpha[i]*fog[i])); cv.drawCircle(float(x[i]),float(y[i]),float(rad[i]*scale_r*f[i]),p)
    if edges is not None and ealpha>0.003:
        p=skia.Paint(AntiAlias=True,StrokeWidth=2.2,Style=skia.Paint.kStroke_Style,StrokeCap=skia.Paint.kRound_Cap)
        for a,b in edges:
            p.setShader(skia.GradientShader.MakeLinear([skia.Point(float(x[a]),float(y[a])),skia.Point(float(x[b]),float(y[b]))],
                 [col4(cols[a],ealpha*min(alpha[a],alpha[b])).toColor(),col4(cols[b],ealpha*min(alpha[a],alpha[b])).toColor()]))
            cv.drawLine(float(x[a]),float(y[a]),float(x[b]),float(y[b]),p)
    p=skia.Paint(AntiAlias=True); pr=skia.Paint(AntiAlias=True,Style=skia.Paint.kStroke_Style,StrokeWidth=1.6)
    for i in order:
        r=float(rad[i]*f[i]); c=cols[i]*fog[i]+(1-fog[i])
        q=p
        if blur is not None and blur[i]>0.5:
            q=skia.Paint(AntiAlias=True,MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle,float(blur[i])))
        q.setColor4f(col4(c,alpha[i])); cv.drawCircle(float(x[i]),float(y[i]),r,q)
        if blur is None or blur[i]<=0.5:
            pr.setColor4f(skia.Color4f(1,1,1,float(0.85*alpha[i]))); cv.drawCircle(float(x[i]),float(y[i]),r,pr)
    for (txt,tx,ty,size,a) in texts:
        if a<=0.003: continue
        font=skia.Font(FONT,size); w=font.measureText(txt)
        pt=skia.Paint(AntiAlias=True); pt.setColor4f(skia.Color4f(0.07,0.07,0.09,float(a)))
        cv.drawString(txt,float(tx-w/2),float(ty),font,pt)
    img=surf.makeImageSnapshot().toarray()  # BGRA? -> handle
    return img
def encode(script,n,out,workers=12):
    import sys
    chunk=int(np.ceil(n/workers)); jobs=[]
    for k,s0 in enumerate(range(0,n,chunk)):
        p=f"{out}.part{k:02d}.mp4"
        jobs.append((p,subprocess.Popen([sys.executable,script,'seg',str(s0),str(min(n,s0+chunk)),p])))
    for p,j in jobs: assert j.wait()==0
    with open(out+'.list','w') as fh:
        for p,_ in jobs: fh.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',out+'.list','-c','copy',out+'.v.mp4'],check=True)
    for p,_ in jobs: os.remove(p)
    os.remove(out+'.list')
def seg(fn,a,b,path):
    pr=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgba' ,'-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-preset','slow','-crf','15','-pix_fmt','yuv420p','-tune','animation',path],stdin=subprocess.PIPE)
    for t in range(a,b):
        img=fn(t); pr.stdin.write(np.ascontiguousarray(img).tobytes())
    pr.stdin.close(); pr.wait()
