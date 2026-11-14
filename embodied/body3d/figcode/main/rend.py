import os, sys, numpy as np
os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, "$BODY_WORK/body3d")
import mujoco, hum as H, env as E
M = H.M; M.vis.global_.offwidth = 3000; M.vis.global_.offheight = 3000; D = mujoco.MjData(M)
BODY = set(H.BODY_GEOMS)
HIDE = [g for g in range(M.ngeom) if g not in BODY]
for g in HIDE: M.geom_rgba[g, 3] = 0
M.site_rgba[:, 3] = 0
def setpose(qpos):
    D.qpos[:] = qpos; mujoco.mj_forward(M, D); return D
def rest():
    mujoco.mj_resetData(M, D); D.qpos[H.JQ] = E.Q0; mujoco.mj_forward(M, D); return D
class Cam:
    def __init__(s, W, Hh, lookat, dist, az, el):
        s.r = mujoco.Renderer(M, Hh, W); s.W, s.H = W, Hh
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE; c.lookat[:] = lookat; c.distance = dist; c.azimuth = az; c.elevation = el; s.c = c
    def render(s, d, dots=None, show=()):
        """dots: list of (points[n,3], rgba, radius)"""
        for g in show: M.geom_rgba[g, 3] = 1
        s.r.update_scene(d, s.c); sc = s.r.scene
        for g in show: M.geom_rgba[g, 3] = 0
        for P, rgba, rad in (dots or []):
            for p in P:
                if sc.ngeom >= sc.maxgeom: break
                g = sc.geoms[sc.ngeom]; mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]), p, np.eye(3).flatten(), np.array(rgba, np.float32)); sc.ngeom += 1
        gc = sc.camera[0]; s.pos = (np.array(gc.pos) + np.array(sc.camera[1].pos)) / 2; s.fwd = np.array(gc.forward); s.up = np.array(gc.up)
        img = s.r.render().copy()
        s.r.enable_segmentation_rendering(); seg = s.r.render().copy(); s.r.disable_segmentation_rendering()
        # seg[...,0] = object id, seg[...,1] = object type (mjOBJ_GEOM=5)
        oid, ot = seg[..., 0], seg[..., 1]
        keep = (oid >= 0) & ~((ot == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(oid, [g for g in HIDE if g not in show]))
        alpha = keep.astype(np.uint8) * 255
        return np.dstack([img, alpha])
    def proj(s, P):
        P = np.atleast_2d(P); right = np.cross(s.fwd, s.up); v = P - s.pos
        z = v @ s.fwd; x = v @ right; y = v @ s.up
        f = s.H / 2 / np.tan(np.deg2rad(M.vis.global_.fovy) / 2)
        return np.stack([s.W / 2 + f * x / z, s.H / 2 - f * y / z], 1)
