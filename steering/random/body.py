# front silhouette, unit box 400x1000; regions clipped to the outline
def cr(pts, closed=True):
    n=len(pts); P=lambda i: pts[i%n]
    d=f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(n if closed else n-1):
        p0,p1,p2,p3=P(i-1),P(i),P(i+1),P(i+2)
        c1=(p1[0]+(p2[0]-p0[0])/6,p1[1]+(p2[1]-p0[1])/6); c2=(p2[0]-(p3[0]-p1[0])/6,p2[1]-(p3[1]-p1[1])/6)
        d+=f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d+(" Z" if closed else "")
R=[(200,136),(222,138),(226,176),(250,188),(292,198),(318,222),(328,262),(334,330),(340,400),(348,480),(354,556),(366,584),(372,622),(364,656),(346,668),(332,650),(328,606),(326,566),(314,480),(306,410),(296,330),(282,266),(266,300),(258,360),(254,420),(262,470),(270,520),(266,600),(254,690),(250,740),(256,800),(250,870),(238,918),(250,948),(252,968),(214,970),(208,930),(214,860),(212,770),(210,700),(208,600),(200,548)]
L=[(400-x,y) for x,y in reversed(R[1:])][1:]  # mirror, drop duplicate centre
OUT=cr(R+L)
HEAD="M200,10 C236,10 258,40 258,80 C258,118 236,150 200,150 C164,150 142,118 142,80 C142,40 164,10 200,10 Z"
def rect(x0,y0,x1,y1): return f"M{x0},{y0} L{x1},{y0} L{x1},{y1} L{x0},{y1} Z"
def both(x0,y0,x1,y1): return rect(x0,y0,x1,y1)+" "+rect(400-x1,y0,400-x0,y1)
REG={  # body regions (before clipping)
 "head": rect(130,0,270,72),
 "face": rect(130,72,270,152),
 "neck": rect(160,140,240,190),
 "shoulders": both(256,175,345,262),
 "chest": rect(140,190,260,335),
 "belly": rect(130,335,270,445),
 "hips": rect(120,445,280,560),
 "arms": both(272,262,360,560),
 "hands": both(300,560,400,700),
 "legs": rect(120,560,280,700)+" "+rect(120,768,280,925),
 "knees": rect(120,700,280,768),
 "feet": rect(120,925,280,1000),
}
def svg(fill, stroke="#2b2b2b"):
    s=f'<defs><clipPath id="bd"><path d="{OUT}"/><path d="{HEAD}"/></clipPath></defs><g clip-path="url(#bd)">'
    order=["legs","knees","feet","hips","belly","chest","arms","hands","shoulders","neck","face","head"]
    for k in order: s+=f'<path d="{REG[k]}" fill="{fill(k)}" stroke="#ffffff" stroke-width="3"/>'
    s+='</g>'
    s+=f'<path d="{OUT}" fill="none" stroke="{stroke}" stroke-width="4" stroke-linejoin="round"/><path d="{HEAD}" fill="none" stroke="{stroke}" stroke-width="4"/>'
    return s
