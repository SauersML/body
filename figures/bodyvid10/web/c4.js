import {setup,smooth,fade} from './common2.js';
// body -> two touch sentences land on their spots -> morph into Qwen3-32B's layout (turning camera)
// -> left and right overlap -> open like a book -> back to the body
export const T=51.0; let A,S,n,DV,LV,V0,B,cB,u,v,w,twins;
export const T0=14.5,DUR=6.0, O0=36.5,OD=4.5, TB=42.0,DURB=5.0;
const S1=38,S2=279;
const YAW0=[[0,-14],[3.6,-6],[8.5,-3],[13,6]];
function key(K,t){ if(t<=K[0][0]) return K[0][1]; for(let i=1;i<K.length;i++){const [a,x]=K[i-1],[b,y]=K[i]; if(t<=b){const s=(t-a)/(b-a); return x+(y-x)*(1-Math.cos(Math.PI*s))/2;}} return K[K.length-1][1]; }
const bb=P=>{const lo=[0,1,2].map(k=>Math.min(...P.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...P.map(p=>p[k])));return lo.map((l,k)=>(l+hi[k])/2);};
// smooth framing center: hard min/max switch between dots and kink the camera path, so use a soft (log-sum-exp) box blended with the mean
const sbb=P=>[0,1,2].map(k=>{const K=0.15; let mx=-1e9,mn=1e9,sm=0; for(const p of P){mx=Math.max(mx,p[k]);mn=Math.min(mn,p[k]);sm+=p[k];}
  let a=0,b=0; for(const p of P){a+=Math.exp(K*(p[k]-mx)); b+=Math.exp(-K*(p[k]-mn));} const hi=mx+Math.log(a)/K, lo=mn-Math.log(b)/K; return 0.5*(hi+lo)*0.6+0.4*sm/P.length;});
const mix=(a,b,s)=>a.map((x,k)=>x*(1-s)+b[k]*s);
export async function init(){ A=await setup(); S=A.S; n=A.n;
  DV=new Float32Array(await (await fetch('dv2.bin')).arrayBuffer()); LV=new Float32Array(await (await fetch('lagv.bin')).arrayBuffer());
  const VC=new Float32Array(await (await fetch('vcol.bin')).arrayBuffer());
  for(let i=0;i<A.col.length;i++) A.col[i]=A.SKIN[i]*0.72+VC[i]*0.28; A.geo.attributes.color.needsUpdate=true; V0=A.pos0;
  B=S.XM.map((p,i)=>p.map((x,k)=>x+S.NS[i][k]*1.6*0.55));
  const mid=S.Y.filter((p,i)=>S.side[i]==0).map(p=>p[0]).sort((a,b)=>a-b); const u0=mid[Math.floor(mid.length/2)];
  u=S.Y.map(p=>p[0]-u0); v=S.Y.map(p=>p[1]); w=S.Y.map(p=>p[2]);
  twins=[]; for(let i=0;i<n;i++) if(S.side[i]>0) twins.push([i,S.part[i]]);
  cB=sbb(B); }
export const amt=(t,l)=> t<TB? smooth((t-T0-l)/DUR) : 1-smooth((t-TB-l)/DURB);
export const open=t=> smooth((t-O0)/OD);
function book(o){ const phi=Math.PI/2-Math.PI*o, sp=Math.sin(phi), cp=Math.cos(phi);
  return [S.Y.map((p,i)=> S.sd[i]<0? [u[i]*sp,v[i],w[i]+u[i]*cp] : [u[i],v[i],w[i]]),sp,cp]; }
function yawAt(t){ if(t<14) return key(YAW0,t);
  const s=smooth((t-14)/19); let y=6-386*s;                 // one full turn around the layout, ending front-left
  y+= -12*smooth((t-40.5)/2.5)+ (8+12)*smooth((t-43.5)/5.0);   // settle to the front for the return to the body
  return y; }
let QL=null;
export function frame(t,fast=false){
  const o=open(t); const [Bk,sp,cp]=book(o);
  const m=S.lag.map(l=>amt(t,l)); const mm=m.reduce((a,b)=>a+b,0)/n;
  const P=A.pos; let ylo=1e9;
  if(!fast) for(let q=0;q<LV.length;q++){ const a=amt(t,LV[q]); const j=3*q;
    let x=V0[j]+a*DV[j], y=V0[j+1]+a*DV[j+1], z=V0[j+2]+a*DV[j+2];
    if(x<0 && a>0){ const uu=-x; x=x*(1-a)+uu*sp*a; z=z+uu*cp*a; }
    P[j]=x; P[j+1]=y; P[j+2]=z; if(y<ylo) ylo=y; }
  if(!fast){ A.geo.attributes.position.needsUpdate=true; A.geo.computeVertexNormals(); A.geo.computeBoundingSphere();
  A.ground.position.y=ylo-0.5; }
  const Q=B.map((b,i)=>b.map((x,k)=>x*(1-m[i])+Bk[i][k]*m[i]));
  const zm=smooth((t-3.4)/1.5)*(1-smooth((t-12.6)/2.0)), s12=smooth((t-8.1)/2.1);
  const tg=mix(B[S1],B[S2],s12); const cT=mix(cB,[tg[0]*0.8,tg[1]+4,tg[2]],zm);
  const c=mix(cT,sbb(Q),mm);
  const so=Math.min(1,Math.max(0,(t-14)/19));
  const yaw=yawAt(t), pitch=9+3*Math.sin(2*Math.PI*t/T)-3*zm+16*Math.sin(Math.PI*smooth(so))*mm;
  A.camera(c,yaw*Math.PI/180,pitch*Math.PI/180,900-300*zm-90*mm,75+40*zm);
  const g=smooth(mm*2.2);
  A.meshAlpha(1-0.32*g, g);
  const f1=fade(t,3.9,8.6,0.7), f2=fade(t,8.9,13.4,0.7), dim=0.72*fade(t,3.8,13.6,0.9);
  const L1=smooth((t-4.5)/1.6), L2=smooth((t-9.5)/1.6);
  const R=new Array(n).fill(1.6), Al=new Array(n).fill(1-dim), Bl=new Array(n).fill(0);
  const pulse=1+0.08*Math.sin(2*Math.PI*1.1*t);
  // each touched spot drops from its sentence onto the skin, then glows while the sentence is shown
  const cy=Math.cos(yaw*Math.PI/180),sy=Math.sin(yaw*Math.PI/180),cpp=Math.cos(pitch*Math.PI/180),spp=Math.sin(pitch*Math.PI/180);
  const up=[spp*sy,cpp,-spp*cy], ez=[-cpp*sy,spp,cpp*cy];
  // the spot leaves its home unseen (fades out), reappears under the sentence, and drops onto the skin
  const drop=(i,L,f,D,a,sw)=>{ const hide=smooth((t-a)/0.5)*(1-smooth((t-a-0.8)/0.5)); const on=t>sw?1-L:0;
    Q[i]=Q[i].map((x,k)=>x+(up[k]*D+ez[k]*35)*on); R[i]=1.6+2.6*f*pulse*(0.6+0.4*L)+1.2*Math.sin(Math.PI*L)*f; Al[i]=(1-dim*(1-f))*(1-hide); };
  drop(S1,L1,f1,78,3.4,4.05); drop(S2,L2,f2,95,8.6,9.2);
  QL={Q,Al}; if(fast){ A.camera(c,yaw*Math.PI/180,pitch*Math.PI/180,900-300*zm-90*mm,75+40*zm); return; }
  A.setSpots(Q,S.cols,R,Al,Bl);
  A.setEdges(Q,twins,S.cols,0.55*fade(t,32.6,43.5,1.6)*mm);
  A.text([["407 places on the body",fade(t,0.5,3.6)],
    ["A feather brushes slowly across the inside",fade(t,4.3,8.5),110,66],["of your left forearm, a hand-width above the wrist.",fade(t,4.3,8.5),196,66],
    ["An ice cube touches your right kneecap,",fade(t,9.3,13.3),110,66],["cold and wet, then slides away.",fade(t,9.3,13.3),196,66],
    ["the same places inside Qwen3-32B",fade(t,18.5,25.0)],["left and right lie on top of each other",fade(t,33.2,37.4),128,74]]);
  A.whiteFade(1-Math.min(1,t/1.0,(T-t)/0.8)); A.render(); }
export function probe(t,ids){ const [Bk]=book(open(t)); return ids.map(i=>{const m=amt(t,S.lag[i]); return B[i].map((x,k)=>x*(1-m)+Bk[i][k]*m);}); }

export function probeS(t,ids){ frame(t,true); A.cam.updateMatrixWorld(); const V=new A.THREE.Vector3();
  return ids.map(i=>{ V.set(...QL.Q[i]).project(A.cam); return [(V.x+1)*960,(1-V.y)*540,QL.Al[i]]; }); }
