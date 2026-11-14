import * as M from './c7.js';
import {fade} from './common7.js';
// v5: the v4 scenes on a faster clock (warp.json: new time -> v4 time) plus a label saying which projection the model's map is.
let WP; export let T=47.8;
const w=t=>{ const i=t/WP.dt, k=Math.min(WP.orig.length-2,Math.max(0,Math.floor(i))), f=i-k; return WP.orig[k]*(1-f)+WP.orig[k+1]*f; };
export async function init(){ WP=await (await fetch('warp.json')).json(); T=WP.T; window.T=T; await M.init(); }
const L1='Qwen3-32B, layer 53: top 4 principal components', L2='rotated to best match the body';
export function frame(tn){ const t=w(tn); M.frame(t);
  const a=Math.max(fade(t,24.0,36.6,0.8),fade(t,46.8,62.6,0.8)); const el=document.getElementById('x');
  el.innerHTML=a<0.003?'':`<div style="position:absolute;left:0;width:1920px;top:962px;text-align:center;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-weight:600;color:#2b2b33;opacity:${a};text-shadow:0 0 8px #fff,0 0 16px #fff;line-height:1.18"><div style="font-size:48px">${L1}</div><div style="font-size:40px;color:#55555f">${L2}</div></div>`; }
