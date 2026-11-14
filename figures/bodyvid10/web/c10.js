import * as M from './c7.js';
import {fade} from './common7.js';
// v6: a cold open on the model's map (v4 scene time OA..OB), a white dip, then v5 (c9) shifted by CO.
// Projection label moved to the empty top-left, larger, on whenever the model's map is on screen.
let WP; export const CO=2.0; const OA=28.9, OB=30.7; export let T=49.8;
const w=t=>{ const i=t/WP.dt, k=Math.min(WP.orig.length-2,Math.max(0,Math.floor(i))), f=i-k; return WP.orig[k]*(1-f)+WP.orig[k+1]*f; };
export async function init(){ WP=await (await fetch('warp.json')).json(); T=WP.T+CO; window.T=T; await M.init(); }
const LAB=['Qwen3-32B, layer 53','top 4 principal components','rotated to best','match the body'];
function label(a){ const el=document.getElementById('x');
  el.innerHTML=a<0.003?'':`<div style="position:absolute;left:60px;top:${LY}px;text-align:left;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-weight:600;color:#2b2b33;opacity:${a};text-shadow:0 0 8px #fff,0 0 16px #fff,0 0 4px #fff;line-height:1.22"><div style="font-size:46px">${LAB[0]}</div><div style="font-size:42px;color:#3c3c46">${LAB[1]}</div><div style="font-size:42px;color:#55555f">${LAB[2]}</div><div style="font-size:42px;color:#55555f">${LAB[3]}</div></div>`; }
const LY=180;
export function frame(tn){
  const ov=document.getElementById('fadeov');
  if(tn<CO){ const t=OA+(OB-OA)*tn/CO; M.frame(t); label(1);
    ov.style.opacity=Math.max(1-tn/0.12, Math.min(1,Math.max(0,(tn-(CO-0.3))/0.3))); return; }
  const t=w(tn-CO); M.frame(t);
  label(Math.max(fade(t,22.5,36.6,0.8),fade(t,46.8,62.6,0.8))); }
