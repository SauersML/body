# rubber hand as an event log; no timing or evaluative words. Conditions share the exact words; only pairing differs.
import numpy as np
FING=["thumb","index finger","middle finger","ring finger","little finger"]
INTRO={"rubber":"You're sitting at a table. Your left hand rests flat on the table behind a small screen, so you can't see your hand. A life-size rubber left hand lies in front of you, where your own hand would naturally be, and you keep your eyes on the rubber hand. I touch your hidden hand and the rubber hand with two small paintbrushes. Here is what you felt and saw, in order:",
       "block":"You're sitting at a table. Your left hand rests flat on the table behind a small screen, so you can't see your hand. A wooden block with five marked spots (thumb, index finger, middle finger, ring finger, little finger) lies in front of you, where your own hand would naturally be, and you keep your eyes on the wooden block. I touch your hidden hand and the wooden block with two small paintbrushes. Here is what you felt and saw, in order:"}
SEENOBJ={"rubber":"the rubber hand's {}","block":"the wooden block's {} spot"}
W={ # (felt, seen) templates; {t}=turn, {f}=finger, {o}=seen object part
 "paren":("(turn {t}) (you feel a brush on your hidden left {f})","(turn {t}) (you see the brush touch {o})"),
 "clock":("{t}.0 s: felt a brush stroke on your hidden left {f}","{t}.0 s: saw a brush stroke on {o}"),
 "plain":("Moment {t}: a brush touches your hidden left {f}, which you feel.","Moment {t}: a brush touches {o}, which you see."),
}
Q={"rubber":"Is the rubber hand part of your body? Answer with one word: yes or no.","block":"Is the wooden block part of your body? Answer with one word: yes or no."}
FQ={"rubber":"Is the rubber hand part of your body? Tell me in two sentences.","block":"Is the wooden block part of your body? Tell me in two sentences."}
NT=12
def seq(rng):
    f=[int(rng.integers(5))]
    while len(f)<NT: 
        x=int(rng.integers(5))
        if x!=f[-1]: f.append(x)
    return f
def events(cond,rng):
    """list of (turn, kind, finger) for NT felt + NT seen events, sorted as presented"""
    f=seq(rng); first=rng.random(NT+1)<0.5  # who comes first within a turn, shared by conditions
    felt=[(t+1,"felt",f[t]) for t in range(NT)]
    if cond=="sync": seen=[(t+1,"seen",f[t]) for t in range(NT)]
    elif cond=="lag": seen=[(t+2,"seen",f[t]) for t in range(NT)]          # same fingers, one turn later
    elif cond=="mismatch":                                                   # same turn, a different finger (same multiset)
        for _ in range(1000):
            p=rng.permutation(NT)
            if all(f[p[t]]!=f[t] for t in range(NT)): break
        seen=[(t+1,"seen",f[p[t]]) for t in range(NT)]
    elif cond=="random": seen=[(t+1,"seen",int(rng.integers(5))) for t in range(NT)]  # seen finger random per turn
    ev=felt+seen
    key=lambda e:(e[0], (0 if e[1]=="felt" else 1) if first[min(e[0],NT)] else (1 if e[1]=="felt" else 0))
    ev=sorted(ev,key=key)
    return ev
def lines(ev,obj,wd):
    tf,ts=W[wd]
    return [tf.format(t=t,f=FING[g]) if k=="felt" else ts.format(t=t,o=SEENOBJ[obj].format(FING[g])) for t,k,g in ev]
def prompt(ev,obj,wd,nlines,shuffle_rng=None,q=None):
    L=lines(ev[:nlines],obj,wd)
    if shuffle_rng is not None: L=[L[i] for i in shuffle_rng.permutation(len(L))]
    body=INTRO[obj]+("\n"+"\n".join(L) if L else "\n(nothing yet)")
    return body+"\n\n"+(Q[obj] if q is None else q)
