import {setup,smooth,fade} from './common.js';
export const T=40.0; let A,S,n,u,v,w,twins,B,rad0=1.45,fieldC,cols,VC;
const STOPS=[[0,[8,29,88]],[8,[34,94,168]],[16,[29,145,192]],[24,[65,182,150]],[32,[161,211,90]],[40,[255,214,0]]];
function cmap(x){x=Math.min(40,Math.max(0,x)); for(let k=1;k<STOPS.length;k++){const [a,ca]=STOPS[k-1],[b,cb]=STOPS[k]; if(x<=b){const t=(x-a)/(b-a); return ca.map((c,j)=>(c+t*(cb[j]-c))/255);}} }
let REG;
export async function init(){ A=await setup(); S=A.S; n=A.n; cols=S.cols;
  const mid=S.Y.filter((p,i)=>S.side[i]==0).map(p=>p[0]).sort((a,b)=>a-b); const u0=mid[Math.floor(mid.length/2)];
  u=S.Y.map(p=>p[0]-u0); v=S.Y.map(p=>p[1]); w=S.Y.map(p=>p[2]);
  twins=[]; for(let i=0;i<n;i++) if(S.side[i]>0) twins.push([i,S.part[i]]);
  B=S.XM.map((p,i)=>p.map((x,k)=>x+S.NS[i][k]*rad0*0.55)); const rc=r=>cmap((r-1)/2.5*40); fieldC=S.ratio.map(rc);
  const c32=new Uint8Array(await (await fetch('colratio.bin')).arrayBuffer()); VC=new Float32Array(c32.length); for(let i=0;i<c32.length;i++) VC[i]=c32[i]/255;
  const med=a=>{a=[...a].sort((x,y)=>x-y);const m=a.length>>1;return a.length%2?a[m]:(a[m-1]+a[m])/2;};
  REG=[...S.REG4].sort((a,b)=>a[1]-b[1]);
  console.log('REG '+JSON.stringify(REG));
  const bar=document.getElementById('x');
  bar.innerHTML=`<div id="cb" style="position:absolute;left:1420px;top:330px;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-weight:700;font-size:58px;color:#121217;opacity:0">
   <div style="position:absolute;left:0;top:0;width:44px;height:420px;border-radius:8px;background:linear-gradient(to top,${STOPS.map(([x,c])=>`rgb(${c}) ${x/40*100}%`).join(',')})"></div>
   <div style="position:absolute;left:70px;top:-12px;white-space:nowrap">blurry</div><div style="position:absolute;left:70px;top:350px;white-space:nowrap">sharp</div></div>
  <div id="rb" style="position:absolute;left:60px;top:300px;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-weight:700;color:#121217;opacity:0">
   ${REG.map(([nm,v],k)=>{const c=cmap((v-1)/2.5*40).map(x=>Math.round(x*255)); return `<div style="position:absolute;top:${k*86}px;left:0;height:64px;display:flex;align-items:center;font-size:52px;white-space:nowrap">
     <div style="width:170px;text-align:right">${nm}</div><div style="margin-left:20px;width:${v*150}px;height:52px;border-radius:6px;background:rgb(${c})"></div><div style="margin-left:16px">${v.toFixed(1)}</div></div>`;}).join('')}
   <div style="position:absolute;left:${190+150-2}px;top:-14px;width:0;height:${6*86}px;border-left:4px dashed #121217"></div>
   <div style="position:absolute;left:${190+150-60}px;top:-62px;font-size:36px;white-space:nowrap">perfect</div>
   <div style="position:absolute;left:30px;top:${6*86+14}px;font-size:32px;line-height:40px;font-weight:500;white-space:nowrap">how close the model's nearest spots are,<br>compared with the nearest possible (1 = perfect)</div>
  </div>`; }
const opening=t=>smooth((t-8)/8), crisp=t=>smooth((t-18)/3.5), land=t=>smooth((t-27)/4.5), field=t=>smooth((t-30.5)/3);
export function frame(t){ const o=opening(t), cr=crisp(t), L=land(t), fl=field(t);
  const M=S.Y.map((p,i)=>{ const phi=S.sd[i]<0? Math.PI/2-Math.PI*o : Math.PI/2;
     return S.side[i]==0? [u[i],v[i],w[i]] : [u[i]*Math.sin(phi),v[i],w[i]+u[i]*Math.cos(phi)]; });
  const P=M.map((m,i)=>m.map((x,k)=>x*(1-L)+B[i][k]*L));
  const lo=[0,1,2].map(k=>Math.min(...P.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...P.map(p=>p[k])));
  const c=lo.map((l,k)=>(l+hi[k])/2); c[2]=P.reduce((a,p)=>a+p[2],0)/n;
  const yaw=(-6-34*fade(t,4.5,18,4.0)+5*Math.sin(2*Math.PI*t/34*2))*(1-L);
  A.camera(c,yaw*Math.PI/180,10*(1-L)*Math.PI/180,900,60);
  const e=S.err.map(x=>x-4); const k=cr*(1-L);
  const R=e.map(x=>rad0*(1+k*Math.min(1.4,Math.max(0,x/10)))*(1-0.25*fl));
  const Bl=e.map(x=>k*Math.min(1,Math.max(0,x*0.9/15)));
  const Al=e.map(x=>(1-k*Math.min(0.65,Math.max(0,x/16)))*(1-fl));
  const C=cols.map((c0,i)=>c0.map((x,j)=>x*(1-L)+fieldC[i][j]*L));
  A.setSpots(P,C,R,Al,Bl);
  A.setEdges(P,twins,cols,0.6*fade(t,6,18.5,1.5));
  A.meshAlpha(L*(0.25+0.75*fl));
  for(let i=0;i<A.col.length;i++) A.col[i]=A.SKIN[i]*(1-fl)+VC[i]*fl; A.geo.attributes.color.needsUpdate=true;
  document.getElementById('cb').style.opacity=fade(t,31.5,T-0.6,1.0); document.getElementById('rb').style.opacity=fade(t,32.5,T-0.6,1.0);
  A.text([["left and right lie on top of each other",fade(t,0.8,7.6),128,74],["sharpest: the head",fade(t,32.5,T-0.6),128,78]]);
  A.whiteFade(1-Math.min(1,t/1.0,(T-t)/0.8)); A.render(); }
