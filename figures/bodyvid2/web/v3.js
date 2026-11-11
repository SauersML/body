import {setup,smooth,fade} from './common.js';
export const T=32.0; let A,DV,VC,V0;
export async function init(){ A=await setup();
  DV=new Float32Array(await (await fetch('dv.bin')).arrayBuffer()); VC=new Float32Array(await (await fetch('vcol.bin')).arrayBuffer());
  A.col.set(VC); A.geo.attributes.color.needsUpdate=true; V0=A.pos0; }
const amt=t=> t<5?0: t<20? smooth((t-5)/7) : 1-smooth((t-20)/7);
export function frame(t){ const m=amt(t); const P=A.pos; let lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];
  for(let i=0;i<P.length;i++){P[i]=V0[i]+m*DV[i]; const k=i%3; if(P[i]<lo[k])lo[k]=P[i]; if(P[i]>hi[k])hi[k]=P[i];}
  A.geo.attributes.position.needsUpdate=true; A.geo.computeVertexNormals(); A.geo.computeBoundingSphere();
  A.ground.position.y=lo[1]-0.5;
  const c=lo.map((l,k)=>(l+hi[k])/2);
  A.camera(c,-28*Math.PI/180*Math.sin(2*Math.PI*t/T),8*Math.PI/180,900,75);
  A.meshAlpha(1); A.setSpots(A.S.XM,A.S.cols,new Array(A.n).fill(0),new Array(A.n).fill(0));
  A.text([["a smooth body surface",fade(t,0.6,4.8)],["the same surface inside Qwen3-32B",fade(t,12.6,19.4)]]);
  A.whiteFade(1-Math.min(1,t/1.0,(T-t)/0.8)); A.render(); }
