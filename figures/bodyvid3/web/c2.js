import {setup,smooth,fade} from './common2.js';
// body -> model (book open) -> fold (left/right overlap) -> open -> back to body
export const T=40.5; let A,S,n,DV,LV,V0,B,Bk,cB,u,v,w,twins;
export const T0=3.5,DUR=7.5,TB=33.5,DURB=4.2,F0=25.0,FD=3.0,O0=30.5,OD=3.5;
const YAW=[[0,-15],[9,22],[18,30],[24,5],[28.5,-36],[32,-34],[36.5,-12],[40.5,-8]];
function key(K,t){ if(t<=K[0][0]) return K[0][1]; for(let i=1;i<K.length;i++){const [a,x]=K[i-1],[b,y]=K[i]; if(t<=b){const s=(t-a)/(b-a); return x+(y-x)*(1-Math.cos(Math.PI*s))/2;}} return K[K.length-1][1]; }
export async function init(){ A=await setup(); S=A.S; n=A.n;
  DV=new Float32Array(await (await fetch('dv2.bin')).arrayBuffer()); LV=new Float32Array(await (await fetch('lagv.bin')).arrayBuffer());
  const VC=new Float32Array(await (await fetch('vcol.bin')).arrayBuffer());
  for(let i=0;i<A.col.length;i++) A.col[i]=A.SKIN[i]*0.72+VC[i]*0.28; A.geo.attributes.color.needsUpdate=true; V0=A.pos0;
  B=S.XM.map((p,i)=>p.map((x,k)=>x+S.NS[i][k]*1.6*0.55));
  const mid=S.Y.filter((p,i)=>S.side[i]==0).map(p=>p[0]).sort((a,b)=>a-b); const u0=mid[Math.floor(mid.length/2)];
  u=S.Y.map(p=>p[0]-u0); v=S.Y.map(p=>p[1]); w=S.Y.map(p=>p[2]);
  twins=[]; for(let i=0;i<n;i++) if(S.side[i]>0) twins.push([i,S.part[i]]);
  cB=bb(B); }
const bb=P=>{const lo=[0,1,2].map(k=>Math.min(...P.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...P.map(p=>p[k])));return lo.map((l,k)=>(l+hi[k])/2);};
export const amt=(t,l)=> t<TB? smooth((t-T0-l)/DUR) : 1-smooth((t-TB-l)/DURB);
export const open=t=> 1-smooth((t-F0)/FD)+smooth((t-O0)/OD);
export function frame(t){
  const o=open(t), phi=Math.PI/2-Math.PI*o, sp=Math.sin(phi), cp=Math.cos(phi);
  const Bk=S.Y.map((p,i)=> S.sd[i]<0? [u[i]*sp,v[i],w[i]+u[i]*cp] : [u[i],v[i],w[i]]);
  const m=S.lag.map(l=>amt(t,l)); const mm=m.reduce((a,b)=>a+b,0)/n;
  const P=A.pos; let ylo=1e9;
  for(let q=0;q<LV.length;q++){ const a=amt(t,LV[q]); const j=3*q;
    let x=V0[j]+a*DV[j], y=V0[j+1]+a*DV[j+1], z=V0[j+2]+a*DV[j+2];
    if(x<0 && o<1){ const uu=-x; x=uu*sp; z=z+uu*cp; }
    P[j]=x; P[j+1]=y; P[j+2]=z; if(y<ylo) ylo=y; }
  A.geo.attributes.position.needsUpdate=true; A.geo.computeVertexNormals(); A.geo.computeBoundingSphere();
  A.ground.position.y=ylo-0.5;
  const Q=B.map((b,i)=>b.map((x,k)=>x*(1-m[i])+Bk[i][k]*m[i]));
  const cK=bb(Bk); const c=cB.map((x,k)=>x*(1-mm)+cK[k]*mm);
  const yaw=key(YAW,t), pitch=10+4*Math.sin(2*Math.PI*t/T);
  A.camera(c,yaw*Math.PI/180,pitch*Math.PI/180,900,75);
  const g=smooth(mm*2.2), fo=1-o;
  A.meshAlpha((1-0.32*g)*(1-0.25*fo), g);
  A.setSpots(Q,S.cols,new Array(n).fill(1.6),new Array(n).fill(1));
  A.setEdges(Q,twins,S.cols,0.55*fade(t,25.3,35.5,1.6)*mm);
  A.text([["407 places on the body",fade(t,0.5,4.6)],["the same places inside Qwen3-32B",fade(t,11,24.6)],["left and right lie on top of each other",fade(t,27,31.8),128,74]]);
  A.whiteFade(1-Math.min(1,t/1.0,(T-t)/0.8)); A.render(); }
export function probe(t,ids){ // dot positions for smoothness checks
  const o=open(t), phi=Math.PI/2-Math.PI*o, sp=Math.sin(phi), cp=Math.cos(phi);
  return ids.map(i=>{const bk=S.sd[i]<0? [u[i]*sp,v[i],w[i]+u[i]*cp] : [u[i],v[i],w[i]]; const m=amt(t,S.lag[i]); return B[i].map((x,k)=>x*(1-m)+bk[k]*m);}); }
