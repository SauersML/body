import {setup,smooth,fade} from './common.js';
export const T=52.0; let A,S,n,DV,LV,V0,B,R,cB,cY;
const T0=5.0,T1=34.0,DUR=8.0;   // out from T0, back from T1
export async function init(){ A=await setup(); S=A.S; n=A.n;
  DV=new Float32Array(await (await fetch(new URLSearchParams(location.search).get('dv')||'dv2.bin')).arrayBuffer()); LV=new Float32Array(await (await fetch('lagv.bin')).arrayBuffer());
  const VC=new Float32Array(await (await fetch('vcol.bin')).arrayBuffer());
  for(let i=0;i<A.col.length;i++) A.col[i]=A.SKIN[i]*0.72+VC[i]*0.28; A.geo.attributes.color.needsUpdate=true; V0=A.pos0;
  const rad=1.6; R=new Array(n).fill(rad);
  B=S.XM.map((p,i)=>p.map((x,k)=>x+S.NS[i][k]*rad*0.55));
  const bb=P=>{const lo=[0,1,2].map(k=>Math.min(...P.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...P.map(p=>p[k])));return lo.map((l,k)=>(l+hi[k])/2);};
  cB=bb(S.XM); cY=bb(S.Yu); }
const amt=(t,l)=> t<T0?0: t<T1? smooth((t-T0-l)/DUR) : 1-smooth((t-T1-l)/DUR);
export function frame(t){
  const m=S.lag.map(l=>amt(t,l)); const mm=m.reduce((a,b)=>a+b,0)/n;
  const P=A.pos; let ylo=1e9;
  for(let v=0;v<LV.length;v++){ const a=amt(t,LV[v]); for(let k=0;k<3;k++){const j=3*v+k; P[j]=V0[j]+a*DV[j];} if(P[3*v+1]<ylo) ylo=P[3*v+1]; }
  A.geo.attributes.position.needsUpdate=true; A.geo.computeVertexNormals(); A.geo.computeBoundingSphere();
  A.ground.position.y=ylo-0.5;
  const Q=B.map((b,i)=>b.map((x,k)=>x*(1-m[i])+S.Yu[i][k]*m[i]));
  const c=cB.map((x,k)=>x*(1-mm)+cY[k]*mm);
  const yaw=38*Math.sin(2*Math.PI*(t-2)/26), pitch=9+5*Math.sin(2*Math.PI*t/T);
  A.camera(c,yaw*Math.PI/180,pitch*Math.PI/180,900,75);
  A.meshAlpha(1-0.32*smooth(mm*2.2));
  A.setSpots(Q,S.cols,R,new Array(n).fill(1));
  A.text([["407 places on the body",fade(t,0.6,5.4)],["the same places inside Qwen3-32B",fade(t,15.5,32.5)]]);
  A.whiteFade(1-Math.min(1,t/1.0,(T-t)/0.8)); A.render(); }
