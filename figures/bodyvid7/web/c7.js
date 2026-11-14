import {setup,smooth,fade} from './common7.js';
// Scene 1: body -> two touch sentences -> the body's two halves pull apart (left | right)
//          -> morph into Qwen3-32B's layout (drawn opened, left beside right) -> back to the body -> halves close.
// Scene 2: the same layout folds shut (left lies on top of right), then opens again like a book.
export const T=63.3;
export const T0=20.0,DUR=6.0, TB=34.0,DURB=5.5, SC2=46.0;
const G=20;                          // how far each half is pulled out
const S1=38,S2=279;
let A,S,n,DV,LV,V0,B,cB,u,v,w,twins,MID;
function key(K,t){ if(t<=K[0][0]) return K[0][1]; for(let i=1;i<K.length;i++){const [a,x]=K[i-1],[b,y]=K[i]; if(t<=b){const s=(t-a)/(b-a); return x+(y-x)*(1-Math.cos(Math.PI*s))/2;}} return K[K.length-1][1]; }
const sbb=P=>[0,1,2].map(k=>{const K=0.15; let mx=-1e9,mn=1e9,sm=0; for(const p of P){mx=Math.max(mx,p[k]);mn=Math.min(mn,p[k]);sm+=p[k];}
  let a=0,b=0; for(const p of P){a+=Math.exp(K*(p[k]-mx)); b+=Math.exp(-K*(p[k]-mn));} const hi=mx+Math.log(a)/K, lo=mn-Math.log(b)/K; return 0.5*(hi+lo)*0.6+0.4*sm/P.length;});
const mix=(a,b,s)=>a.map((x,k)=>x*(1-s)+b[k]*s);
export async function init(){ A=await setup(); S=A.S; n=A.n;
  DV=new Float32Array(await (await fetch('dv2.bin')).arrayBuffer()); LV=new Float32Array(await (await fetch('lagv.bin')).arrayBuffer());
  MID=new Float32Array(await (await fetch('mid.bin')).arrayBuffer());
  const VC=new Float32Array(await (await fetch('vcol.bin')).arrayBuffer());
  for(let i=0;i<A.col.length;i++) A.col[i]=A.SKIN[i]*0.72+VC[i]*0.28; A.geo.attributes.color.needsUpdate=true; V0=A.pos0;
  B=S.XM.map((p,i)=>p.map((x,k)=>x+S.NS[i][k]*1.6*0.55));
  const mid=S.Y.filter((p,i)=>S.side[i]==0).map(p=>p[0]).sort((a,b)=>a-b); const u0=mid[Math.floor(mid.length/2)];
  u=S.Y.map(p=>p[0]-u0); v=S.Y.map(p=>p[1]); w=S.Y.map(p=>p[2]);
  twins=[]; for(let i=0;i<n;i++) if(S.side[i]>0) twins.push([i,S.part[i]]);
  cB=sbb(B); }
export const split=t=> t<TB+3? smooth((t-14.4)/3.2) : 1-smooth((t-41.0)/2.8);
export const amt=(t,l)=> t>=SC2? 1 : t<TB? smooth((t-T0-l)/DUR) : 1-smooth((t-TB-l)/DURB);
export const open=t=> t<SC2? 1 : 1-smooth((t-48.0)/4.5)+smooth((t-55.5)/5.0);
function book(o){ const phi=Math.PI/2-Math.PI*o, sp=Math.sin(phi), cp=Math.cos(phi);
  return [S.Y.map((p,i)=> S.sd[i]<0? [u[i]*sp,v[i],w[i]+u[i]*cp] : [u[i],v[i],w[i]]),sp,cp]; }
const YAW1=[[0,-14],[3.6,-6],[8.5,-3],[13,4],[16.5,0],[20,0],[27,-28],[33.5,26],[40,0],[45.5,-6]];
const YAW2=[[SC2,-38],[52,-30],[T,-16]];
let QL=null;
export function frame(t,fast=false){
  const s2=t>=SC2, sp1=s2?0:split(t);
  const o=open(t); const [Bk,sp,cp]=book(o);
  const m=S.lag.map(l=>amt(t,l)); const mm=m.reduce((a,b)=>a+b,0)/n;
  const P=A.pos; let ylo=1e9;
  if(!fast){ for(let q=0;q<LV.length;q++){ const a=amt(t,LV[q]); const j=3*q;
    const x0=V0[j]; const off=(x0>0?1:-1)*G*sp1*(1-MID[q]);
    let x=x0+a*DV[j]+(1-a)*off, y=V0[j+1]+a*DV[j+1], z=V0[j+2]+a*DV[j+2];
    if(x<0 && a>0){ const uu=-x; x=x*(1-a)+uu*sp*a; z=z+uu*cp*a; }
    P[j]=x; P[j+1]=y; P[j+2]=z; if(y<ylo) ylo=y; }
    A.geo.attributes.position.needsUpdate=true; A.geo.computeVertexNormals(); A.geo.computeBoundingSphere();
    A.ground.position.y=ylo-0.5; A.CUT.value=sp1; }
  const Bs=B.map((b,i)=>[b[0]+S.sd[i]*G*sp1,b[1],b[2]]);
  const Q=Bs.map((b,i)=>b.map((x,k)=>x*(1-m[i])+Bk[i][k]*m[i]));
  let c,yaw,pitch,dist,offy,zm=0;
  if(!s2){
    zm=smooth((t-3.4)/1.5)*(1-smooth((t-12.6)/2.0)); const s12=smooth((t-8.1)/2.1);
    const tg=mix(B[S1],B[S2],s12); const cT=mix(cB,[tg[0]*0.8,tg[1]+4,tg[2]],zm);
    c=mix(cT,sbb(Q),mm); yaw=key(YAW1,t); pitch=9+3*Math.sin(2*Math.PI*t/T)-3*zm+8*mm;
    dist=900-300*zm-60*mm; offy=75+40*zm;
  } else { c=sbb(Q); yaw=key(YAW2,t); pitch=15+3*Math.sin(Math.PI*(t-SC2)/(T-SC2)); dist=880; offy=75; }
  if(fast){ A.camera(c,yaw*Math.PI/180,pitch*Math.PI/180,dist,offy); QL={Q,Al:new Array(n).fill(1)}; return; }
  A.camera(c,yaw*Math.PI/180,pitch*Math.PI/180,dist,offy);
  const g=smooth(mm*2.2); A.meshAlpha(1-0.32*g, g);
  const f1=fade(t,3.9,8.6,0.7), f2=fade(t,8.9,13.4,0.7), dim=0.72*fade(t,3.8,13.6,0.9);
  const L1=smooth((t-4.5)/1.6), L2=smooth((t-9.5)/1.6);
  const R=new Array(n).fill(1.6), Al=new Array(n).fill(1-dim), Bl=new Array(n).fill(0);
  if(!s2){
    const pulse=1+0.08*Math.sin(2*Math.PI*1.1*t);
    const cy=Math.cos(yaw*Math.PI/180),sy=Math.sin(yaw*Math.PI/180),cpp=Math.cos(pitch*Math.PI/180),spp=Math.sin(pitch*Math.PI/180);
    const up=[spp*sy,cpp,-spp*cy], ez=[-cpp*sy,spp,cpp*cy];
    const drop=(i,L,f,D,a,sw)=>{ const hide=smooth((t-a)/0.5)*(1-smooth((t-a-0.8)/0.5)); const on=t>sw?1-L:0;
      Q[i]=Q[i].map((x,k)=>x+(up[k]*D+ez[k]*35)*on); R[i]=1.6+2.6*f*pulse*(0.6+0.4*L)+1.2*Math.sin(Math.PI*L)*f; Al[i]=(1-dim*(1-f))*(1-hide); };
    drop(S1,L1,f1,78,3.4,4.05); drop(S2,L2,f2,95,8.6,9.2);
  }
  QL={Q,Al};
  A.setSpots(Q,S.cols,R,Al,Bl);
  A.setEdges(Q,twins,S.cols,s2? 0.5*fade(t,50.2,62.6,1.8):0);
  const lr=s2?0:fade(t,15.6,28.0,1.0); let xr=470,xl=1450,yr=560,yl=560;
  if(lr>0.003){ A.cam.updateMatrixWorld(); const V=new A.THREE.Vector3(); let mnx=1e9,mxx=-1e9,sy0=0,sy1=0,c0=0,c1=0;
    for(let i=0;i<n;i++){ if(S.sd[i]==0) continue; V.set(...Q[i]).project(A.cam); const X=(V.x+1)*960,Y=(1-V.y)*540;
      if(S.sd[i]<0){ mnx=Math.min(mnx,X); if(S.cat[i]=='hand'){sy0+=Y;c0++;} } else { mxx=Math.max(mxx,X); if(S.cat[i]=='hand'){sy1+=Y;c1++;} } }
    xr=mnx-120; xl=mxx+105; yr=sy0/c0+28; yl=sy1/c1+28; }
  A.text2([["407 places on the body",fade(t,0.5,3.6)],
    ["A feather brushes slowly across the inside",fade(t,4.3,8.5),960,110,66],["of your left forearm, a hand-width above the wrist.",fade(t,4.3,8.5),960,196,66],
    ["An ice cube touches your right kneecap,",fade(t,9.3,13.3),960,110,66],["cold and wet, then slides away.",fade(t,9.3,13.3),960,196,66],
    ["right",lr,xr,yr,84],["left",lr,xl,yl,84],
    ["the same places inside Qwen3-32B",fade(t,23.5,31.5)],
    ["left and right lie on top of each other",fade(t,51.6,57.6,0.9),960,128,74]]);
  const wd=smooth(1-Math.abs(t-SC2)/1.1);
  A.whiteFade(Math.max(wd,1-Math.min(1,t/1.0,(T-t)/0.8))); A.render(); }
export function probeS(t,ids){ frame(t,true); A.cam.updateMatrixWorld(); const V=new A.THREE.Vector3();
  return ids.map(i=>{ V.set(...QL.Q[i]).project(A.cam); return [(V.x+1)*960,(1-V.y)*540]; }); }
