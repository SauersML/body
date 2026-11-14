import numpy as np, mujoco, hum as H, env as E
from scipy.sparse import csr_matrix; from scipy.sparse.csgraph import shortest_path
d = mujoco.MjData(H.M); d.qpos[H.JQ] = E.Q0; mujoco.mj_forward(H.M, d); P, N = H.taxel_world(d)
B = H.M.geom_bodyid[H.TG]; par = H.M.body_parentid
adj = (B[:, None] == B[None]) | (par[B][:, None] == B[None]) | (B[:, None] == par[B][None])
D = np.linalg.norm(P[:, None] - P[None], axis=-1); W = np.where(adj & (D < 0.07), D, 0)
G = shortest_path(csr_matrix(W), directed=False); print("inf frac", np.isinf(G).mean(), "max", G[np.isfinite(G)].max())
np.save("geo.npy", G.astype(np.float32)); np.save("taxel_rest.npy", P)
