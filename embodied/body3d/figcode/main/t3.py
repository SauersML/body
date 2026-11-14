from rend import *
from PIL import Image
import pickle
V=pickle.load(open('$BODY_WORK/body3d/vids_expert.pkl','rb'))
tr=V['poke']['traj'][0]; qpos, mp, mq, sk = tr[28]
D.qpos[:]=qpos; D.mocap_pos[:]=mp; D.mocap_quat[:]=mq; mujoco.mj_forward(M,D)
P,N=H.taxel_world(D); sk=sk.astype(float); part=H.TPART==1
hot=(sk>0.05)&part
c = Cam(900, 1200, [0, 0, 1.15], 2.2, 160, -5)
im=c.render(D, dots=[(P[~hot]+N[~hot]*0.003,[0.15,0.15,0.15,1],0.005),(P[hot]+N[hot]*0.004,[0.82,0.26,0.18,1],0.009)], show=(32,))
Image.fromarray(im,'RGBA').save('poke.png'); print(hot.sum(), set(H.TPART[hot]))
