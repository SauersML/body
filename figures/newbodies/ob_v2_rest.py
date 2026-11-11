# legend line under panels
ly=1350
x=L0+6
ax.add_patch(Circle((x+16,ly),16,fc=ACC,ec="white",zorder=5)); x=runs(x+44,ly,[("new body part",ACC,"semibold")],px=36)+60
ax.add_patch(Circle((x+16,ly),14,fc="white",ec=INK,lw=4.5*PX,zorder=5)); x=runs(x+44,ly,[("human part touched in the same chat",INK,"medium")],px=36)+60
ax.add_patch(Circle((x+10,ly),8,fc=HUM,zorder=5)); runs(x+34,ly,[("54 human places, each read out with its own place left out of training",MID,"regular")],px=36)
# ---------- nearest-neighbour tables
ty=1440
T(L0,ty,"Closest human places to each new part",px=42,w="semibold",fam="Inter Display")
T(L0+760,ty+2,"(cosine similarity of layer-52 activations, 1 = identical)",px=36,c=MID)
ROWS={"robot":[("left wheel","wheelL"),("right wheel","wheelR"),("gripper","gripper"),("camera eye","camera"),("antenna","antenna"),("chassis","chassis")],
      "octopus":[("arm 1","arm1"),("arm 5","arm5"),("suckers","suckers"),("arm tip","armtip"),("mantle","mantle"),("eye","eye")],
      "tail":[("tail base","tailbase"),("tail tip","tailtip"),("top of head","crown"),("lower back","lowback")]}
NEAR={}
for i,b in enumerate(["robot","octopus","tail"]):
    x=L0+pw*i+6; y=ty+78
    for nm,p in ROWS[b]:
        t3=top3(b,p); NEAR[f"{b}:{p}"]=[(a_,round(float(v),2)) for a_,v in t3]
        ctl=p in ("crown","lowback")
        T(x,y,nm,px=38,c=INK if ctl else ACC,w="semibold")
        xx=x
        for a_,v in t3:
            t=T(xx,y+44,a_,px=36,c=INK); xx+=wid(t)+8; t=T(xx,y+44,f"{v:.2f}"[1:],px=36,c=MID); xx+=wid(t)+30
        y+=100
        if b=="tail" and p=="tailtip":
            y+=20; T(x,y,"human parts in the same chat, for scale:",px=34,c=MID); y+=56
# ---------- steering
sy=2150
ax.plot([L0,R0],[sy-40,sy-40],color=HUML,lw=3*PX)
T(L0,sy+10,"Adding a human-part direction at layer 52 makes Qwen3-32B name the matching robot part",px=46,w="semibold",fam="Inter Display")
cy=sy+80
box(L0,cy,1115,108); T(L0+28,cy+30,"user, robot chat",px=32,c=MID,w="medium")
T(L0+28,cy+76,"\u201cIs there any spot on your body where you notice a sensation?\u201d",px=36)
def steer_ex(y,dirn,quote_parts):
    box(L0,y,1115,56,fc="white",ec=ACC,lw=4,r=16); T(L0+24,y+28,"+ "+dirn,px=36,c=ACC,w="semibold")
    box(L0,y+76,1115,150,fc=CARD); T(L0+28,y+76+30,"Qwen3-32B",px=32,c=MID,w="medium")
    yy=y+76+80
    for line in quote_parts:
        runs(L0+28,yy,line,px=36); yy+=44
steer_ex(cy+136,"human left foot direction",[[("\u201cIn my ",INK,"regular"),("left wheel",ACC,"semibold"),(", I notice a slight pressure and",INK,"regular")],[("coolness from being in contact with the floor.\u201d",INK,"regular")]])
steer_ex(cy+390,"human top of head direction",[[("\u201cAh, the top of my ",INK,"regular"),("camera eye",ACC,"semibold"),(" is the one drawing my",INK,"regular")],[("attention \u2026\u201d",INK,"regular")]])
# bars
bx=L0+1170; bw=R0-bx-150; n=24
def k(key,word): return round(sc[key][word]*n)
G=[("human left foot","wheel","human left foot"),("human top of head","camera eye","human head top"),("human left hand","gripper","human left hand")]
WORD={"wheel":"wheel","camera eye":"camera/eye","gripper":"gripper"}
STEER={}
lx=bx
for c,lab in [(ACC,"human-part direction added"),(RND,"random direction"),(NONE,"no steering")]:
    ax.add_patch(Rectangle((lx,cy+2),34,34,fc=c,ec="none")); t=T(lx+46,cy+19,lab,px=34,c=INK); lx+=46+wid(t)+40
y=cy+70
for gi,(hn,word,dk) in enumerate(G):
    vals=[(k(f"body_robot|{dk}|1.0",WORD[word]),ACC,"human-part direction added"),(k("body_robot|random|1.0",WORD[word]),RND,"random direction added"),(k("body_robot|none|0",WORD[word]),NONE,"no steering")]
    STEER[f"{hn} -> {word}"]=[v for v,_,_ in vals]
    runs(bx,y+18,[(hn+" direction  \u2192  says \u201c",INK,"semibold"),(word,ACC,"semibold"),("\u201d",INK,"semibold")],px=38)
    yb=y+60
    for v,c,lab in vals:
        ax.add_patch(Rectangle((bx,yb),max(bw*v/n,3),40,fc=c,ec="none",zorder=2))
        T(bx+bw*v/n+14,yb+20,f"{v}/24",px=36,c=INK,w="semibold" if c==ACC else "regular")
        yb+=48
    y=yb+30
ax.plot([bx,bx],[cy+120,y-30],color=INK,lw=2.5*PX,zorder=3)
T(bx,y+4,"replies (out of 24) naming that robot part",px=34,c=MID)
json.dump({"near":NEAR,"steer_k_of_24":STEER,"readout_R2_heldout":dict(zip(["|x|","height","side"],r2.round(2).tolist())),
           "readout_pos":{b:{p:[round(float(z[0]),1),round(float(z[1]),1)] for p,z in ZN[b].items()} for b in ZN}},open(CF+"fig/other_bodies_v2_numbers.json","w"),indent=1)
