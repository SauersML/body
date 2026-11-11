import {setup,smooth,fade} from './common.js';
export const T=34.0; let A;
const order={head:0,arm:1,hand:2,torso:1.5,leg:2.5,foot:3};
let B,Y,lag,cB,cY,R,cols;
export async function init(){ A=await setup(); const S=A.S; const n=A.n; cols=S.cols;
  const rad=1.45; R=new Array(n).fill(rad);
  B=S.XM.map((p,i)=>p.map((x,k)=>x+S.NS[i][k]*rad*0.55)); Y=S.Y;
  const ymax=Math.max(...B.map(p=>p[1]));
  lag=S.XM.map((p,i)=>order[S.cat[i]]/3*1.2+(ymax-p[1])/177*0.3);
  const bb=P=>{const lo=[0,1,2].map(k=>Math.min(...P.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...P.map(p=>p[k])));return lo.map((l,k)=>(l+hi[k])/2);};
  cB=bb(A.S.XM); cY=bb(Y); }
function amt(t){ return lag.map(l=> t<4.5?0: t<20? smooth((t-4.5-l)/5.5) : 1-smooth((t-20-l)/5.5)); }
export function frame(t){ const m=amt(t); const mm=m.reduce((a,b)=>a+b,0)/m.length;
  const P=B.map((b,i)=>b.map((x,k)=>x*(1-m[i])+Y[i][k]*m[i]));
  const c=cB.map((x,k)=>x*(1-mm)+cY[k]*mm);
  A.camera(c,32*Math.PI/180*Math.sin(2*Math.PI*t/T),9*Math.PI/180,900,75);
  A.meshAlpha(1-0.82*smooth(mm*1.6));
  A.setSpots(P,cols,R,new Array(A.n).fill(1));
  A.setEdges(P,A.S.E,cols,0.55*fade(t,13,24,2.5));
  A.text([["407 places on the body",fade(t,0.6,4.6)],["the same places inside Qwen3-32B",fade(t,12.6,19.6)]]);
  A.whiteFade(1-Math.min(1,t/1.0,(T-t)/0.8)); A.render(); }
