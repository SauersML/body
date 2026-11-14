"""Renders for body3d_steer_v3 panel A: rest body with skin sensors, chest-touch body, all-touch body, plain body."""
import os, numpy as np
from PIL import Image
from rend import *
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
M.light_castshadow[:] = 0
POKES = np.load('pokes400.npy')
DOT = [0.13, 0.13, 0.13, 1]; REDF = [0.82, 0.26, 0.18, 1]
sim = E.Sim("mount", np.random.default_rng(0)); q0 = sim.d.qpos.copy()

def crop_alpha(im, pad=6):
    a = np.array(im)[..., 3]; ys, xs = np.where(a > 0)
    return (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)

def body(q, red_mask, cam, red_rad=0.011, dot_rad=0.0045):
    setpose(q); Pw, Nw = H.taxel_world(D)
    dots = [(Pw[~red_mask] + Nw[~red_mask] * 0.003, DOT, dot_rad)]
    if red_mask.any(): dots.append((Pw[red_mask] + Nw[red_mask] * 0.004, REDF, red_rad))
    return Image.fromarray(cam.render(D, dots=dots), 'RGBA')

cam = Cam(900, 1300, [0, 0, 0.98], 2.55, 180, -3)
chest = H.PARTS.index('chest')
m_chest = np.zeros(H.NT, bool); m_chest[POKES[POKES[:, 1] == chest, 0]] = True
m_all = np.zeros(H.NT, bool); m_all[POKES[:, 0]] = True
none = np.zeros(H.NT, bool)
import pickle
qm = pickle.load(open('$BODY_RUNS/steer_confirm_Qwen3-8B_named.pkl', 'rb'))['runs']['none'][0]['qpos_final']
ims = dict(moved=body(qm, none, cam), sensors=body(q0, none, cam, dot_rad=0.006), chest=body(q0, m_chest, cam), all=body(q0, m_all, cam), plain=body(q0, none, cam))
box = crop_alpha(ims['all'])
for k, im in ims.items(): im.crop(box if k != 'moved' else crop_alpha(im)).save(f'a3_{k}.png')
np.save('a3_q0.npy', q0)
print('n_chest', int(m_chest.sum()), int((POKES[:, 1] == chest).sum()), 'n_all', len(POKES), 'NT', H.NT, 'NJ', E.NJ)
