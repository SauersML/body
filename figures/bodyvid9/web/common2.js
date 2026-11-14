import * as THREE from 'three';
import {RoomEnvironment} from './jsm/environments/RoomEnvironment.js';
export {THREE};
export const W=1920,H=1080;
const bin=async(f,T)=>new T(await (await fetch(f)).arrayBuffer());
export const smooth=x=>{x=Math.min(1,Math.max(0,x));return x*x*x*(x*(6*x-15)+10);};
export const fade=(t,a,b,r=0.7)=>Math.min(1,Math.max(0,Math.min((t-a)/r,(b-t)/r)));
export async function setup(){
  const S=await (await fetch('spots.json')).json();
  const ren=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true,preserveDrawingBuffer:true});
  ren.setPixelRatio(window.DPR||1); ren.setSize(W,H); ren.setClearColor(0xffffff,1);
  ren.outputColorSpace=THREE.SRGBColorSpace; ren.shadowMap.enabled=true; ren.shadowMap.type=THREE.PCFSoftShadowMap;
  const scene=new THREE.Scene(); const pm=new THREE.PMREMGenerator(ren); scene.environment=pm.fromScene(new RoomEnvironment(ren),0.04).texture;
  // ---- real mesh
  const pos0=await bin('pos.bin',Float32Array), idx=await bin('idx.bin',Uint32Array), nrm=await bin('nrm.bin',Float32Array);
  const geo=new THREE.BufferGeometry();
  const pos=new Float32Array(pos0); geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('normal',new THREE.BufferAttribute(new Float32Array(nrm),3)); geo.setIndex(new THREE.BufferAttribute(idx,1));
  const nv=pos0.length/3; const col=new Float32Array(nv*3).fill(0); for(let i=0;i<nv;i++){col[3*i]=0.90;col[3*i+1]=0.885;col[3*i+2]=0.87;}
  const SKIN=new Float32Array(col); geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  const mat=new THREE.MeshPhysicalMaterial({vertexColors:true,roughness:0.55,metalness:0,sheen:0.25,sheenRoughness:0.6,sheenColor:new THREE.Color(1,1,1),envMapIntensity:0.35,transparent:true,side:THREE.DoubleSide});
  mat.onBeforeCompile=s=>{s.fragmentShader=s.fragmentShader.replace('#include <color_fragment>','#include <color_fragment>\n diffuseColor.rgb=pow(diffuseColor.rgb,vec3(2.2));');};
  const mesh=new THREE.Mesh(geo,mat); mesh.castShadow=true; mesh.renderOrder=0; scene.add(mesh);
  mat.depthWrite=false;
  // depth-only prepass: the skin always occludes spots the same way, whatever its opacity
  const dmat=new THREE.MeshBasicMaterial({colorWrite:false,side:THREE.DoubleSide}); const dmesh=new THREE.Mesh(geo,dmat); dmesh.renderOrder=-1; dmesh.frustumCulled=false; scene.add(dmesh);
  const key=new THREE.DirectionalLight(0xffffff,1.3); key.castShadow=true; key.shadow.mapSize.set(2048,2048); key.shadow.radius=8; key.shadow.blurSamples=16; key.shadow.bias=-0.0005;
  key.position.set(-120,300,300); key.target.position.set(0,90,0); const sc=key.shadow.camera; sc.left=-150;sc.right=150;sc.top=150;sc.bottom=-150;sc.near=1;sc.far=800;
  scene.add(key); scene.add(key.target); scene.add(new THREE.HemisphereLight(0xffffff,0xd9d4cf,0.6));
  const ground=new THREE.Mesh(new THREE.PlaneGeometry(1200,1200),new THREE.ShadowMaterial({opacity:0.10,transparent:true}));
  ground.rotation.x=-Math.PI/2; ground.position.y=-0.5; ground.receiveShadow=true; scene.add(ground);
  // ---- spots: lit sphere impostors with optional blur
  const n=S.XM.length; const q=new THREE.InstancedBufferGeometry();
  q.setAttribute('position',new THREE.BufferAttribute(new Float32Array([-1,-1,0, 1,-1,0, 1,1,0, -1,1,0]),3)); q.setIndex([0,1,2,0,2,3]);
  const aP=new THREE.InstancedBufferAttribute(new Float32Array(n*3),3), aC=new THREE.InstancedBufferAttribute(new Float32Array(n*3),3), aR=new THREE.InstancedBufferAttribute(new Float32Array(n),1), aB=new THREE.InstancedBufferAttribute(new Float32Array(n),1), aA=new THREE.InstancedBufferAttribute(new Float32Array(n),1);
  for(const [k,a] of [['iP',aP],['iC',aC],['iR',aR],['iB',aB],['iA',aA]]){a.setUsage(THREE.DynamicDrawUsage); q.setAttribute(k,a);} q.instanceCount=n;
  const smat=new THREE.ShaderMaterial({transparent:true,depthWrite:true,uniforms:{gA:{value:1.0}},
   vertexShader:`attribute vec3 iP,iC; attribute float iR,iB,iA; varying vec2 vUv; varying vec3 vC,vVP; varying float vR,vB,vA;
    void main(){ vec4 mv=modelViewMatrix*vec4(iP,1.0); float ext=1.0+2.5*iB; vUv=position.xy*ext; vec4 p=mv; p.xy+=position.xy*iR*ext;
     vC=iC; vR=iR; vB=iB; vA=iA; vVP=mv.xyz; gl_Position=projectionMatrix*p; }`,
   fragmentShader:`varying vec2 vUv; varying vec3 vC,vVP; varying float vR,vB,vA; uniform mat4 projectionMatrix; uniform float gA;
    void main(){ float r=length(vUv); 
     float aa=fwidth(r);
     float sharpA=1.0-smoothstep(1.0-aa,1.0+aa,r);
     float sig=1.0+2.0*vB; float blurA=exp(-r*r/(2.0*sig*sig*0.35));
     float a=mix(sharpA,blurA,clamp(vB*3.0,0.0,1.0))*vA*gA; if(a<0.004) discard;
     float rr=min(r,1.0); vec3 nrm=vec3(vUv/max(r,1.0),sqrt(max(0.0,1.0-rr*rr)));
     vec3 L=normalize(vec3(-0.45,0.65,0.62)); float dif=max(dot(nrm,L),0.0); vec3 h=normalize(L+vec3(0,0,1)); float sp=pow(max(dot(nrm,h),0.0),40.0);
     vec3 lit=vC*(0.45+0.6*dif)+0.25*sp; 
     float ring=smoothstep(0.80,0.88,r)*(1.0-smoothstep(0.98,1.0,r));
     lit=mix(lit,vec3(1.0),0.55*ring);
     vec3 c=mix(lit,vC,clamp(vB*3.0,0.0,1.0));
     vec4 cp=projectionMatrix*vec4(vVP+vec3(0,0,nrm.z*vR*(1.0-clamp(vB*3.0,0.,1.))),1.0); gl_FragDepth=0.5*(cp.z/cp.w)+0.5;
     gl_FragColor=vec4(pow(c,vec3(1.0/1.0)),a);
     #include <colorspace_fragment>
    }`});
  smat.extensions={fragDepth:true,derivatives:true};
  const spots=new THREE.Mesh(q,smat); spots.frustumCulled=false; spots.renderOrder=3; scene.add(spots);
  // ghost pass: spots hidden behind the skin, shown with a continuous opacity
  const gmat=smat.clone(); gmat.uniforms={gA:{value:0}}; gmat.depthFunc=THREE.GreaterDepth; gmat.depthWrite=false; gmat.extensions={fragDepth:true,derivatives:true};
  const ghost=new THREE.Mesh(q,gmat); ghost.frustumCulled=false; ghost.renderOrder=2.5; scene.add(ghost);
  // ---- edges: screen-space ribbons
  const eg=new THREE.InstancedBufferGeometry();
  eg.setAttribute('position',new THREE.BufferAttribute(new Float32Array([0,-1,0, 1,-1,0, 1,1,0, 0,1,0]),3)); eg.setIndex([0,1,2,0,2,3]);
  const mE=4000; const eA=new THREE.InstancedBufferAttribute(new Float32Array(mE*3),3),eB=new THREE.InstancedBufferAttribute(new Float32Array(mE*3),3),eCA=new THREE.InstancedBufferAttribute(new Float32Array(mE*4),4),eCB=new THREE.InstancedBufferAttribute(new Float32Array(mE*4),4);
  for(const [k,a] of [['eA',eA],['eB',eB],['eCA',eCA],['eCB',eCB]]){a.setUsage(THREE.DynamicDrawUsage); eg.setAttribute(k,a);} eg.instanceCount=0;
  const emat=new THREE.ShaderMaterial({transparent:true,depthWrite:false,uniforms:{res:{value:new THREE.Vector2(W,H)},lw:{value:2.4}},
   vertexShader:`attribute vec3 eA,eB; attribute vec4 eCA,eCB; uniform vec2 res; uniform float lw; varying vec4 vc;
    void main(){ vec4 a=projectionMatrix*modelViewMatrix*vec4(eA,1.0), b=projectionMatrix*modelViewMatrix*vec4(eB,1.0);
     vec2 sa=a.xy/a.w*res*0.5, sb=b.xy/b.w*res*0.5; vec2 d=normalize(sb-sa+1e-6); vec2 nn=vec2(-d.y,d.x);
     vec4 p=mix(a,b,position.x); vec2 off=nn*position.y*lw*0.5/(res*0.5)*p.w; p.xy+=off; vc=mix(eCA,eCB,position.x); gl_Position=p; }`,
   fragmentShader:`varying vec4 vc; void main(){ gl_FragColor=vc; 
     #include <colorspace_fragment>
   }`});
  const edges=new THREE.Mesh(eg,emat); edges.frustumCulled=false; edges.renderOrder=2; scene.add(edges);
  const cam=new THREE.PerspectiveCamera(15.5,W/H,10,4000);
  // sRGB->linear for colors passed to shaders (output converts back)
  const lin=c=>Math.pow(c,2.2);
  const api={THREE,S,n,ren,scene,cam,mesh,mat,geo,pos0,pos,col,SKIN,key,ground,
    setSpots(P,C,R,A,B){ // P: array of [x,y,z]; sorted back to front
      const cp=cam.position; const ord=[...Array(n).keys()].map(i=>[i,(P[i][0]-cp.x)**2+(P[i][1]-cp.y)**2+(P[i][2]-cp.z)**2]).sort((a,b)=>b[1]-a[1]);
      ord.forEach(([i],k)=>{aP.array.set(P[i],3*k); aC.array.set(C[i].map(lin),3*k); aR.array[k]=R[i]; aA.array[k]=A[i]; aB.array[k]=B?B[i]:0;});
      for(const a of [aP,aC,aR,aA,aB]) a.needsUpdate=true;},
    setEdges(P,list,C,alpha){ let k=0; for(const [a,b] of list){ eA.array.set(P[a],3*k); eB.array.set(P[b],3*k);
        eCA.array.set([...C[a].map(lin),alpha],4*k); eCB.array.set([...C[b].map(lin),alpha],4*k); k++; }
      eg.instanceCount=alpha>0.003?k:0; for(const a of [eA,eB,eCA,eCB]) a.needsUpdate=true;},
    camera(c,yaw,pitch,dist=900,offy=40){ // same convention as the old renderer: Q=(P-c)Rx(pitch)Ry(yaw)^T, viewer on +z
      const cy=Math.cos(yaw),sy=Math.sin(yaw),cpp=Math.cos(pitch),sp=Math.sin(pitch);
      // R=Rx@Ry ; camera axis in world = R^T e_z
      const R=[[cy,0,sy],[sp*sy,cpp,-sp*cy],[-cpp*sy,sp,cpp*cy]];
      const ez=[R[2][0],R[2][1],R[2][2]], ey=[R[1][0],R[1][1],R[1][2]];
      cam.position.set(c[0]+dist*ez[0],c[1]+dist*ez[1],c[2]+dist*ez[2]); cam.up.set(...ey); cam.lookAt(c[0],c[1],c[2]);
      cam.setViewOffset(W,H,0,-offy,W,H); cam.updateProjectionMatrix();
      const L=[-0.45,0.65,0.62]; const lw=[0,1,2].map(j=>R[0][j]*L[0]+R[1][j]*L[1]+R[2][j]*L[2]);
      key.position.set(c[0]+400*lw[0],c[1]+400*lw[1],c[2]+400*lw[2]); key.target.position.set(...c); },
    meshAlpha(a,g=0){ mat.opacity=a; mesh.visible=a>0.001; dmesh.visible=a>0.001; ground.material.opacity=0.10*Math.min(1,a); gmat.uniforms.gA.value=g; ghost.visible=g>0.002; },
    whiteFade(a){ document.getElementById('fadeov').style.opacity=a; },
    text(items){ const el=document.getElementById('t'); el.innerHTML=items.filter(x=>x[1]>0.003).map(([s,a,y=128,sz=78])=>`<div style="position:absolute;left:0;right:0;top:${y-sz*0.82}px;font-size:${sz}px;opacity:${a}">${s}</div>`).join(''); },
    render(){ ren.render(scene,cam); }
  };
  return api;
}
