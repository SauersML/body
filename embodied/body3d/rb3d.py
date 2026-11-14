"""Still renders of the humanoid for figures: white background via depth mask, taxel dots, 3-D path tubes, part highlight."""
import os
os.environ.setdefault("MUJOCO_GL", "cgl")
import numpy as np, mujoco
import hum as H, env as E
m = H.M
m.mat_texid[:] = -1  # flat colours, no checker/cross marks
m.geom_group[m.geom("floor").id] = 5
for t in range(m.ntex):
    if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX: m.tex_data[m.tex_adr[t]:m.tex_adr[t] + m.tex_height[t] * m.tex_width[t] * m.tex_nchannel[t]] = 255
m.vis.rgba.haze[:] = [1, 1, 1, 1]
BODY = np.array([0.86, 0.84, 0.82, 1.0])
for g in range(m.ngeom):
    if m.geom(g).name != "floor": m.geom_matid[g] = -1; m.geom_rgba[g] = BODY
m.vis.headlight.ambient[:] = .45; m.vis.headlight.diffuse[:] = .55; m.vis.headlight.specular[:] = .05
m.vis.quality.offsamples = 8
m.light_castshadow[:] = 0

def fk(qpos, prep=None):
    d = mujoco.MjData(m); d.qpos[:] = qpos
    if prep: prep(d)
    mujoco.mj_forward(m, d); return d

def tips(qpos):
    d = fk(qpos); return {h: d.site(f"tip_{h}").xpos.copy() for h in ("left", "right")}

def _add(sc, typ, size, pos, mat, rgba):
    if sc.ngeom >= sc.maxgeom: return None
    g = sc.geoms[sc.ngeom]; mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).flatten(), np.asarray(rgba, np.float32))
    sc.ngeom += 1; return g

def render(qpos, W=1200, Hh=1200, az=150, el=-10, dist=2.4, lookat=(0.1, 0, 0.95), taxels=None, highlight=None,
           paths=(), spheres=(), body_rgba=None, fovy=None, prep=None):
    """taxels: (rgba per taxel or None for all, size); highlight: (part index, rgba); paths: [(Nx3 points, rgba, radius)]"""
    if body_rgba is not None:
        for g in range(1, m.ngeom): m.geom_rgba[g] = body_rgba
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, Hh)
    if fovy: m.vis.global_.fovy = fovy
    r = mujoco.Renderer(m, Hh, W); d = fk(qpos, prep); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, cam); sc = r.scene; P, N = H.taxel_world(d)
    if highlight is not None:
        p, col, rad = highlight
        for i in np.where(H.TPART == p)[0]: _add(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [rad, 0, 0], P[i] + N[i] * 0.002, np.eye(3), col)
    if taxels is not None:
        cols, rad = taxels
        for i in range(H.NT):
            c = cols[i] if cols is not None else [0.2, 0.2, 0.25, 1]
            if c is None: continue
            _add(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [rad if np.ndim(rad) == 0 else rad[i], 0, 0], P[i] + N[i] * 0.003, np.eye(3), c)
    for pts, col, rad in paths:
        for a, b in zip(pts[:-1], pts[1:]):
            g = _add(sc, mujoco.mjtGeom.mjGEOM_CAPSULE, [rad, 0, 0], np.zeros(3), np.eye(3), col)
            if g is not None: mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, rad, np.asarray(a, float), np.asarray(b, float))
    for pos, col, rad in spheres: _add(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [rad, 0, 0], pos, np.eye(3), col)
    rgb = r.render(); r.enable_depth_rendering(); r.update_scene(d, cam)
    dep = r.render(); r.disable_depth_rendering(); r.close()
    # depth pass lacks the added geoms; mask = anything not sky-coloured or nearer than far plane
    bg = dep >= dep.max() * 0.999
    out = rgb.copy(); sky = bg & (np.abs(rgb.astype(int) - rgb[0, 0].astype(int)).sum(-1) < 40)
    out[sky] = 255
    if body_rgba is not None:
        for g in range(1, m.ngeom): m.geom_rgba[g] = BODY
    return out

def smooth(pts, k=6):
    """Catmull-Rom upsampling of a polyline (k points per segment)."""
    pts = np.asarray(pts); P = np.vstack([pts[:1], pts, pts[-1:]]); out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, k, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1]); return np.array(out)
