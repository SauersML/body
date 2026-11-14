import * as THREE from 'three';
import {RoomEnvironment} from './jsm/environments/RoomEnvironment.js';
const W=1920,H=1080, HOLD=0.9, MORPH=1.0, INTRO=4.0, OUTRO=4.0;
const bin=async(f,T)=>new T(await (await fetch(f)).arrayBuffer());
const sm=x=>{x=Math.min(1,Math.max(0,x));return x*x*x*(x*(6*x-15)+10);};
export let T; let S,N,geo,pos,P0,ski,skw,nv,spots=[],ren,scene,cam,ref,lab1,lab2,bar,grp,refgrp;
const COL={arm:[0.16,0.45,0.85],hand:[0.47,0.73,1.0],leg:[0.85,0.33,0.10],foot:[1.0,0.66,0.30],head:[0.20,0.66,0.38],torso:[0.62,0.55,0.75]};
const REGS=['arm','hand','leg','foot','head','torso'];
function col(c,side){const k=COL[c]; const f=side<0?0.72:1.0; return k.map(x=>x*f+(1-f)*0.15);}
export async function init(){
  S=await (await fetch('stages.json')).json(); N=S.Y.length; T=INTRO+N*(HOLD+MORPH)-MORPH+OUTRO;
  ren=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true});
  ren.setSize(W,H); ren.setClearColor(0xffffff,1); ren.outputColorSpace=THREE.SRGBColorSpace;
  scene=new THREE.Scene(); const pm=new THREE.PMREMGenerator(ren); scene.environment=pm.fromScene(new RoomEnvironment(ren),0.04).texture; scene.environmentIntensity=0.35;
  scene.add(new THREE.HemisphereLight(0xffffff,0x888888,0.6)); const dl=new THREE.DirectionalLight(0xffffff,1.6); dl.position.set(-2,4,5); scene.add(dl);
  cam=new THREE.PerspectiveCamera(30,W/H,0.1,100); cam.position.set(0,0,9.5); cam.lookAt(0,0,0);
  P0=await bin('pos.bin',Float32Array); const idx=await bin('idx.bin',Uint32Array); ski=await bin('sk_i.bin',Uint16Array); skw=await bin('sk_w.bin',Float32Array);
  const vreg=await bin('vreg.bin',Uint8Array), vside=await bin('vside.bin',Int8Array); nv=P0.length/3;
  const vc=new Float32Array(nv*3); for(let i=0;i<nv;i++){const c=col(REGS[vreg[i]],vside[i]).map(x=>Math.pow(x,2.2)); vc.set(c,3*i);}
  const mk=(p)=>{const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(p,3)); g.setAttribute('color',new THREE.BufferAttribute(vc,3)); g.setIndex(new THREE.BufferAttribute(idx,1)); g.computeVertexNormals(); return g;};
  const mat=new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.55,metalness:0.0,side:THREE.DoubleSide});
  pos=new Float32Array(P0); geo=mk(pos); grp=new THREE.Group(); grp.add(new THREE.Mesh(geo,mat)); grp.position.set(-1.2,-0.15,0); scene.add(grp);
  const sg=new THREE.SphereGeometry(0.075,24,16);
  for(let k=0;k<S.cats.length;k++){const m=new THREE.Mesh(sg,new THREE.MeshStandardMaterial({color:new THREE.Color().setRGB(...col(S.cats[k],S.side[k]),THREE.SRGBColorSpace),roughness:0.35})); grp.add(m); spots.push(m);}
  refgrp=new THREE.Group(); refgrp.add(new THREE.Mesh(mk(new Float32Array(P0)),mat)); refgrp.scale.setScalar(0.42); refgrp.position.set(3.15,-1.05,0); scene.add(refgrp);
  const t=document.getElementById('t');
  t.innerHTML=`<div id="l1" style="position:absolute;left:1180px;top:70px;width:700px;text-align:left;font-size:64px;color:#55555c">x</div>
   <div id="l2" style="position:absolute;left:1180px;top:150px;width:740px;text-align:left;font-size:80px;line-height:1.0">x</div>
   <div style="position:absolute;left:60px;top:40px;width:1100px;text-align:left;font-size:52px;line-height:1.15">OLMo-2 7B: where its states put<br>touches on each body part</div>
   <div style="position:absolute;left:1390px;top:930px;width:440px;font-size:40px;color:#55555c">real body</div>
   <div id="bar" style="position:absolute;left:60px;top:1030px;height:14px;width:0px;background:#121217;border-radius:7px"></div>`;
  lab1=document.getElementById('l1'); lab2=document.getElementById('l2'); bar=document.getElementById('bar');
}
function place(Y,ang){
  for(let i=0;i<nv;i++){let x=0,y=0,z=0,bx=0,by=0,bz=0; for(let j=0;j<6;j++){const k=ski[6*i+j],w=skw[6*i+j]; x+=w*Y[k][0];y+=w*Y[k][1];z+=w*Y[k][2]; const X=S.XM[k]; bx+=w*X[0];by+=w*X[1];bz+=w*X[2];}
    const a=0.8; pos[3*i]=x+a*(P0[3*i]-bx); pos[3*i+1]=y+a*(P0[3*i+1]-by); pos[3*i+2]=z+a*(P0[3*i+2]-bz);}
  geo.attributes.position.needsUpdate=true; geo.computeVertexNormals();
  for(let k=0;k<spots.length;k++) spots[k].position.set(...Y[k]);
  grp.rotation.y=ang; refgrp.rotation.y=ang;
}
export function frame(t){
  const u=Math.max(0,t-INTRO); let s=Math.floor(u/(HOLD+MORPH)); let f=(u-s*(HOLD+MORPH)-HOLD)/MORPH; if(s>=N-1){s=N-1;f=0;}
  const a=S.Y[s], b=S.Y[Math.min(s+1,N-1)], e=sm(f);
  const Y=a.map((p,k)=>p.map((x,j)=>(1-e)*x+e*b[k][j]).map(v=>v*1.25));
  place(Y,0.5*Math.sin(t*2*Math.PI/24));
  const L=S.labels[f>0.5?Math.min(s+1,N-1):s]; lab1.textContent=L[0]; lab2.textContent=L[1];
  bar.style.width=(1800*Math.min(1,(s+Math.max(0,e))/(N-1)))+'px';
  ren.render(scene,cam);
}
