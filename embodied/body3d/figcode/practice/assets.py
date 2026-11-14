import sys; sys.path.insert(0, '$BODY_WORK/fig/body3d')
import pickle, numpy as np
from PIL import Image
from rend import *
M.light_castshadow[:] = 0
V = pickle.load(open('$BODY_RUNS/selfplay_vids_Qwen3-8B_named_ctrl.pkl', 'rb'))
cam = Cam(900, 1300, [0, 0, 0.98], 2.55, 160, -3)
DOT = [0.13, 0.13, 0.13, 1]; REDF = [0.82, 0.26, 0.18, 1]
def frame(key, j, t, show=(), name='x'):
    qpos, mp, mq, sk = V[key]['traj'][j][t]
    D.qpos[:] = qpos; D.mocap_pos[:] = mp; D.mocap_quat[:] = mq; mujoco.mj_forward(M, D)
    Pw, Nw = H.taxel_world(D); red = sk > 0.05
    dots = [(Pw[~red] + Nw[~red] * 0.003, DOT, 0.0045), (Pw[red] + Nw[red] * 0.004, REDF, 0.011)]
    im = Image.fromarray(cam.render(D, dots=dots, show=show), 'RGBA'); a = np.array(im)[..., 3]; ys, xs = np.where(a > 0)
    im.crop((xs.min() - 6, ys.min() - 6, xs.max() + 6, ys.max() + 6)).save(name + '.png')
for key, ok in (('r4_reach', 1), ('r4_grasp', 1)):
    v = V[key]; j = [i for i, o in enumerate(v['ok']) if o][0]; print(key, j, v['info'][j].get('part'))
    frame(key, j, -1, show=(33, 34), name=key)
