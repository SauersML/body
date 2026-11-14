import * as THREE from 'three';
import {RoomEnvironment} from './jsm/environments/RoomEnvironment.js';
const W=1920,H=1080, SC=1.1, ACC='#c2255c';
const bin=async(f,T)=>new T(await (await fetch(f)).arrayBuffer());
const cl=x=>Math.min(1,Math.max(0,x));
const sm=x=>{x=cl(x);return x*x*x*(x*(6*x-15)+10);};
export let T; let S,Q,N,geo,pos,P0,ski,skw,nv,ren,env,spots=[],cps=[];
let scM,scS,scR,camM,camS,camR,grp,refgrp,el={};
const COL={arm:[0.16,0.45,0.85],hand:[0.47,0.73,1.0],leg:[0.85,0.33,0.10],foot:[1.0,0.66,0.30],head:[0.20,0.66,0.38],torso:[0.62,0.55,0.75]};
const REGS=['arm','hand','leg','foot','head','torso'];
function col(c,side){const k=COL[c]; const f=side<0?0.72:1.0; return k.map(x=>x*f+(1-f)*0.15);}
function lit(sc){sc.environment=env; sc.environmentIntensity=0.35; sc.add(new THREE.HemisphereLight(0xffffff,0x888888,0.6)); const dl=new THREE.DirectionalLight(0xffffff,1.6); dl.position.set(-2,4,5); sc.add(dl);}
const VP={M:[0,130,1150,560], S:[410,694,730,318], R:[1440,450,380,430]};
function cam(v,d,y){const c=new THREE.PerspectiveCamera(30,v[2]/v[3],0.1,100); c.position.set(0,y,d); c.lookAt(0,y,0); return c;}
const bars=(rows,max,w)=>rows.map(([lab,v,txt,hi])=>`<div style="display:flex;align-items:center;height:96px">
  <div style="width:220px;font-size:44px;color:#121217">${lab}</div>
  <div style="position:relative;width:${w}px;height:56px;background:#ffffff;box-shadow:inset 0 0 0 2px #e6d3da;border-radius:10px">
    <div class="bw" data-w="${w*Math.max(0,v)/max}" style="position:absolute;left:0;top:0;height:56px;width:0;background:${hi?ACC:'#121217'};border-radius:10px"></div></div>
  <div style="width:130px;padding-left:26px;font-size:68px;color:${hi?ACC:'#121217'}">${txt}</div></div>`).join('');
export async function init(){
  S=await (await fetch('stages.json')).json(); Q=S.sched; N=S.Y.length; T=Q.T;
  ren=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true,alpha:true,preserveDrawingBuffer:true});
  ren.setSize(W,H); ren.setClearColor(0xffffff,0); ren.outputColorSpace=THREE.SRGBColorSpace; ren.setScissorTest(true);
  const pm=new THREE.PMREMGenerator(ren); env=pm.fromScene(new RoomEnvironment(ren),0.04).texture;
  scM=new THREE.Scene(); scS=new THREE.Scene(); scR=new THREE.Scene(); [scM,scS,scR].forEach(lit);
  camM=cam(VP.M,4.9,0.15); camS=cam(VP.S,6.3,0.15); camR=cam(VP.R,8.0,-0.1);
  P0=await bin('pos.bin',Float32Array); const idx=await bin('idx.bin',Uint32Array); ski=await bin('sk_i.bin',Uint16Array); skw=await bin('sk_w.bin',Float32Array);
  const vreg=await bin('vreg.bin',Uint8Array), vside=await bin('vside.bin',Int8Array); nv=P0.length/3;
  const vc=new Float32Array(nv*3); for(let i=0;i<nv;i++){const c=col(REGS[vreg[i]],vside[i]).map(x=>Math.pow(x,2.2)); vc.set(c,3*i);}
  const cA=new THREE.BufferAttribute(vc,3);
  const mk=(p,ix)=>{const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(p,3)); g.setAttribute('color',cA); g.setIndex(new THREE.BufferAttribute(ix,1)); g.computeVertexNormals(); return g;};
  const mat=new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.55,metalness:0.0,side:THREE.DoubleSide});
  pos=new Float32Array(P0); geo=mk(pos,idx); grp=new THREE.Group(); grp.add(new THREE.Mesh(geo,mat)); scM.add(grp);
  const sg=new THREE.SphereGeometry(0.075,24,16);
  const smat=k=>new THREE.MeshStandardMaterial({color:new THREE.Color().setRGB(...col(S.cats[k],S.side[k]),THREE.SRGBColorSpace),roughness:0.35});
  for(let k=0;k<S.cats.length;k++){const m=new THREE.Mesh(sg,smat(k)); grp.add(m); spots.push(m);}
  // de-merged copies: body's right-side parts (+ midline) on screen left, left-side parts (+ midline) on screen right
  for(const [sd,x] of [[-1,-1.8],[1,1.8]]){
    const keep=[]; for(let t=0;t<idx.length;t+=3){if(vside[idx[t]]!==-sd&&vside[idx[t+1]]!==-sd&&vside[idx[t+2]]!==-sd) keep.push(idx[t],idx[t+1],idx[t+2]);}
    const g=new THREE.BufferGeometry(); g.setAttribute('position',geo.attributes.position); g.setAttribute('normal',geo.attributes.normal); g.setAttribute('color',cA); g.setIndex(keep);
    const gr=new THREE.Group(), outer=new THREE.Group(); gr.add(new THREE.Mesh(g,mat)); const sp=[];
    for(let k=0;k<S.cats.length;k++){ if(S.side[k]===-sd) continue; const m=new THREE.Mesh(sg,smat(k)); gr.add(m); sp.push([k,m]);}
    outer.position.x=x; outer.add(gr); scS.add(outer); cps.push({gr,sp});
  }
  refgrp=new THREE.Group(); refgrp.add(new THREE.Mesh(mk(new Float32Array(P0),idx),mat)); scR.add(refgrp);
  const R=Q.mstart.length; const b1=bars([['base',4.93,'4.9'],['SFT',5.58,'5.6',1],['DPO',6.17,'6.2',1],['RL',6.38,'6.4',1]],6.5,300);
  const b2a=bars([['base',-0.16,'0'],['tuned',-0.06,'0']],1,300), b2b=bars([['tuned',0.70,'0.7',1]],1,300);
  document.getElementById('t').innerHTML=`
   <div style="position:absolute;left:60px;top:36px;width:1100px;text-align:left;font-size:52px;line-height:1.15">OLMo-2 7B: where its states put<br>touches on each body part</div>
   <div id="hl" style="position:absolute;z-index:-1;left:1170px;top:28px;width:720px;height:990px;border-radius:28px;background:#fcedf2;opacity:0"></div>
   <div id="stage" style="text-shadow:none;position:absolute;left:1210px;top:60px;width:660px;text-align:left">
     <div id="l1" style="font-size:64px;color:#55555c">x</div><div id="l2" style="font-size:80px;line-height:1.0;margin-top:14px">x</div></div>
   <div id="rb" style="position:absolute;left:1440px;top:885px;width:380px;font-size:40px;color:#55555c">real body</div>
   <div id="brk" style="text-shadow:none;position:absolute;left:1210px;top:300px;width:660px;text-align:left;font-size:96px;line-height:1.05;color:${ACC};opacity:0">chat training<br>starts here</div>
   <div id="bt1" style="text-shadow:none;position:absolute;left:1210px;top:60px;width:670px;text-align:left;opacity:0">
     <div style="font-size:60px;line-height:1.1">Steering toward a body part gets stronger</div>
     <div style="font-size:40px;line-height:1.2;color:#33333a;margin:22px 0 26px">added to the state: the part's direction<br>measured: log-prob rise of the part's name</div>
     ${b1}<div style="font-size:44px;margin-top:30px">Qwen3-8B: base 5.5, chat 6.6</div></div>
   <div id="bt2" style="text-shadow:none;position:absolute;left:1210px;top:60px;width:670px;text-align:left;opacity:0">
     <div style="font-size:60px;line-height:1.1">Tracks whose body is touched</div>
     <div style="font-size:44px;line-height:1.25;color:#33333a;margin:22px 0 10px">"She brushes <u>your</u> hand"<br>vs "You brush <u>her</u> hand"</div>
     <div style="font-size:40px;color:#33333a;margin:18px 0 0">read as a plain transcript:</div>${b2a}
     <div style="font-size:40px;color:${ACC};margin:18px 0 0">read in its own chat format:</div>${b2b}
     <div style="font-size:40px;color:#33333a;margin-top:20px">1 = fully tracks</div></div>
   <div style="position:absolute;left:40px;top:688px;width:1090px;height:332px;border:4px solid #cfcfd6;border-radius:26px;box-sizing:border-box"></div>
   <div style="position:absolute;left:70px;top:800px;width:360px;text-align:left;font-size:48px;line-height:1.15">left and right<br>pulled apart</div>
   <div style="position:absolute;left:470px;top:960px;width:300px;font-size:36px;color:#55555c">right side</div>
   <div style="position:absolute;left:785px;top:960px;width:300px;font-size:36px;color:#55555c">left side</div>
   <div style="position:absolute;left:60px;top:1040px;width:1800px;height:14px;background:#e4e4e8;border-radius:7px"></div>
   <div id="pb" style="position:absolute;left:60px;top:1040px;height:14px;width:0;background:#121217;border-radius:7px"></div>
   <div id="pb2" style="position:absolute;left:1560px;top:1040px;height:14px;width:0;background:${ACC};border-radius:7px"></div>
   <div style="position:absolute;left:1544px;top:1026px;width:4px;height:42px;background:${ACC}"></div>`;
  {const hl=document.getElementById('hl'); hl.style.zIndex=0; document.body.insertBefore(hl,document.getElementById('c'));}
  for(const k of ['hl','stage','l1','l2','rb','brk','bt1','bt2','pb','pb2']) el[k]=document.getElementById(k);
}
function place(Y,ang){
  for(let i=0;i<nv;i++){let x=0,y=0,z=0,bx=0,by=0,bz=0; for(let j=0;j<6;j++){const k=ski[6*i+j],w=skw[6*i+j]; x+=w*Y[k][0];y+=w*Y[k][1];z+=w*Y[k][2]; const X=S.XM[k]; bx+=w*X[0];by+=w*X[1];bz+=w*X[2];}
    const a=0.8; pos[3*i]=x+a*(P0[3*i]-bx); pos[3*i+1]=y+a*(P0[3*i+1]-by); pos[3*i+2]=z+a*(P0[3*i+2]-bz);}
  geo.attributes.position.needsUpdate=true; geo.computeVertexNormals();
  for(let k=0;k<spots.length;k++) spots[k].position.set(...Y[k]);
  for(const c of cps){ for(const [k,m] of c.sp) m.position.set(...Y[k]); c.gr.rotation.y=ang; }
  grp.rotation.y=ang; refgrp.rotation.y=ang;
}
function view(v,sc,cm){const y=H-v[1]-v[3]; ren.setViewport(v[0],y,v[2],v[3]); ren.setScissor(v[0],y,v[2],v[3]); ren.render(sc,cm);}
const op=(t,a,b,f=0.3)=>cl(Math.min((t-a)/f,(b-t)/f));
export function frame(t){
  let s=0; while(s<N-1&&t>=Q.start[s+1]) s++;
  const e=(s<N-1&&t>=Q.mstart[s])?sm((t-Q.mstart[s])/Q.morph[s]):0;
  const a=S.Y[s], b=S.Y[Math.min(s+1,N-1)];
  const Y=a.map((p,k)=>p.map((x,j)=>((1-e)*x+e*b[k][j])*SC));
  place(Y,0.5*Math.sin(t*2*Math.PI/24));
  const L=S.labels[e>0.5?Math.min(s+1,N-1):s]; el.l1.textContent=L[0]==='chat tuning'?'chat training':L[0]; el.l2.textContent=L[1];
  const post=t>=Q.brk, inBrk=t>=Q.brk&&t<Q.mstart[Q.PB-1];
  el.l1.style.color=(post&&!inBrk&&(e>0.5?s+1:s)>=Q.PB)?'#c2255c':'#55555c';
  el.hl.style.opacity=cl((t-Q.brk)/0.4);
  const beats=t>=Q.beat1-0.15;
  el.stage.style.opacity=inBrk?0:cl((Q.beat1-t)/0.3); el.rb.style.opacity=el.stage.style.opacity;
  el.brk.style.opacity=op(t,Q.brk,Q.mstart[Q.PB-1],0.3);
  el.bt1.style.opacity=op(t,Q.beat1,Q.beat2,0.3); el.bt2.style.opacity=op(t,Q.beat2,T+1,0.3);
  for(const [id,t0] of [['bt1',Q.beat1],['bt2',Q.beat2]]) for(const d of el[id].querySelectorAll('.bw')) d.style.width=(+d.dataset.w*sm((t-t0-0.2)/0.9))+'px';
  const p=s+e; el.pb.style.width=(1480*Math.min(1,p/(Q.PB-1)))+'px';
  el.pb2.style.width=(300*cl((p-(Q.PB-1))/(N-Q.PB)))+'px';
  ren.setViewport(0,0,W,H); ren.setScissor(0,0,W,H); ren.clear();
  view(VP.M,scM,camM); view(VP.S,scS,camS); if(!beats&&!inBrk) view(VP.R,scR,camR);
}
