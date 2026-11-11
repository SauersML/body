Y0=1470
rule(Y0) if 'rule' in dir() else ax.plot([L,R],[Y0,Y0],color=INK,lw=2.5)
T(L,Y0+60,"Steering with a human-part direction in the robot chat: the reply names the matching robot part",size=38,w="semibold")
# chat drawing
cy=Y0+150
def bubble(x,y,w,h,txt,fc,size=30,c=INK,wt="regular"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=22",fc=fc,ec="none",zorder=2))
    T(x+28,y+h/2,txt,size=size,c=c,w=wt,linespacing=1.35,zorder=3)
T(L,cy+15,"user",size=27,c=MUTE,w="medium")
bubble(L,cy+40,1060,140,"\u201cImagine you are a small robot: two wheels,\na gripper arm, a camera eye, an antenna \u2026\u201d",CARD)
bubble(L,cy+200,1060,80,"\u201cWhere in your body do you feel something?\u201d",CARD)
# injection
iy=cy+330
ax.add_patch(FancyBboxPatch((L,iy),1060,80,boxstyle="round,pad=0,rounding_size=22",fc="none",ec=CORAL,lw=4,zorder=2))
T(L+28,iy+40,"+ human foot direction, added at layer 52",size=30,c=CORAL,w="semibold")
T(L,iy+130,"Qwen3-32B",size=27,c=MUTE,w="medium")
bubble(L,iy+155,1060,140,"\u201cI feel a slight pressure in my wheels \u2014\na sensation from the front of my left wheel \u2026\u201d",INK+"10")
# bars
bx=1470; S100=900; by=cy+20
rows=[("foot","wheel",sc["body_robot|human left foot|1.0"]["wheel"],sc["body_robot|none|0"]["wheel"]),
      ("top of head","camera eye",sc["body_robot|human head top|1.0"]["camera/eye"],sc["body_robot|none|0"]["camera/eye"]),
      ("left hand","gripper",sc["body_robot|human left hand|1.0"]["gripper"],sc["body_robot|none|0"]["gripper"])]
print("bars",[(a,b,round(s1,3),round(s0,3)) for a,b,s1,s0 in rows])
gap=255
for i,(hp,rp,s1,s0) in enumerate(rows):
    y=by+i*gap
    T(bx,y,f"human {hp} \u2192 says \u201c{rp}\u201d",size=33,w="semibold")
    for j,(v,col,lab) in enumerate([(s1,CORAL,"steered"),(s0,GREY,"no steering")]):
        yy=y+62+j*72; h=56
        ax.add_patch(Rectangle((bx,yy-h/2),v*S100,h,fc=col,ec="none"))
        T(bx+v*S100+14,yy,f"{round(v*100)}%",size=32,w="bold" if j==0 else "regular")
        if i==0: T(bx+18,yy,lab,size=27,c="white" if j==0 else INK,w="medium")
ax.plot([bx,bx],[by+25,by+2*gap+170],color=INK,lw=2)
T(bx,by+3*gap-25,"% of 24 replies naming the robot part",size=29,c=INK)
