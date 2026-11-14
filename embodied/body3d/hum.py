"""Humanoid with hands, dense skin, sensors. x forward, y left, z up."""
import numpy as np, mujoco, os

def _arm(side):
    s = 1 if side == "left" else -1; L = side
    def fing(name, pos, axis, l1, l2, rng, dr=(0, 0, -1)):
        dr = np.array(dr, float); dr /= np.linalg.norm(dr); e1 = " ".join(f"{x:.4f}" for x in dr * l1); e2 = " ".join(f"{x:.4f}" for x in dr * l2)
        p2 = " ".join(f"{x:.4f}" for x in dr * (l1 + 0.004)); tp = " ".join(f"{x:.4f}" for x in dr * (l2 + 0.01))
        return f"""
          <body name="{name}1_{L}" pos="{pos}">
            <joint name="{name}1_{L}" axis="{axis}" range="{rng}" class="finger"/>
            <geom name="{name}1_{L}" fromto="0 0 0 {e1}" size=".011"/>
            <body name="{name}2_{L}" pos="{p2}">
              <joint name="{name}2_{L}" axis="{axis}" range="0 95" class="finger"/>
              <geom name="{name}2_{L}" fromto="0 0 0 {e2}" size=".010"/>
              {'<site name="tip_'+L+'" pos="'+tp+'" size=".008"/>' if name=='index' else ''}
            </body>
          </body>"""
    ax = f"-{s} 0 0" if s > 0 else "1 0 0"   # finger flexion curls toward body midline
    return f"""
      <body name="upper_arm_{L}" pos="0 {s*0.2} .04">
        <joint name="shoulder_x_{L}" axis="{s} 0 0" range="-10 160" class="arm"/>
        <joint name="shoulder_y_{L}" axis="0 1 0" range="-175 60" class="arm"/>
        <joint name="shoulder_z_{L}" axis="0 0 {s}" range="-90 90" class="arm"/>
        <geom name="upper_arm_{L}" fromto="0 0 -.02 0 0 -.27" size=".04"/>
        <body name="forearm_{L}" pos="0 0 -.30">
          <joint name="elbow_{L}" axis="0 1 0" range="-150 0" class="elbow"/>
          <geom name="forearm_{L}" fromto="0 0 -.01 0 0 -.23" size=".031"/>
          <body name="hand_{L}" pos="0 0 -.26">
            <joint name="wrist_y_{L}" axis="0 1 0" range="-70 70" class="wrist"/>
            <joint name="wrist_x_{L}" axis="1 0 0" range="-40 40" class="wrist"/>
            <geom name="palm_{L}" type="box" size=".012 .035 .045" pos="0 0 -.045" euler="0 0 90"/>
            <site name="palm_{L}" pos="0 {-s*0.03} -.05" size=".01"/>
            <site name="grip_{L}" pos="0 {-s*0.054} -.078" size=".005"/>
            {fing('index', '.02 0 -.092', ax, 0.045, 0.035, '0 95')}
            {fing('middle', '-.022 0 -.092', ax, 0.048, 0.037, '0 95')}
            {fing('thumb', f'.042 {-s*0.012} -.03', f"{-s*0.707:.3f} 0 {-s*0.707:.3f}", 0.035, 0.03, '-20 80', (1, 0, -1))}
          </body>
        </body>
      </body>"""

def _leg(side):
    s = 1 if side == "left" else -1; L = side
    return f"""
          <body name="thigh_{L}" pos="0 {s*0.1} -.04">
            <joint name="hip_x_{L}" axis="{s} 0 0" range="-30 40" class="big"/>
            <joint name="hip_z_{L}" axis="0 0 {s}" range="-40 40" class="big"/>
            <joint name="hip_y_{L}" axis="0 1 0" range="-130 20" class="big"/>
            <geom name="thigh_{L}" fromto="0 0 0 0 {s*0.01} -.34" size=".06"/>
            <body name="shin_{L}" pos="0 {s*0.01} -.4">
              <joint name="knee_{L}" pos="0 0 .02" axis="0 -1 0" range="-150 0" class="big"/>
              <geom name="shin_{L}" fromto="0 0 0 0 0 -.3" size=".049"/>
              <body name="foot_{L}" pos="0 0 -.39">
                <joint name="ankle_y_{L}" pos="0 0 .08" axis="0 1 0" range="-50 50" class="ankle"/>
                <joint name="ankle_x_{L}" pos="0 0 .04" axis="1 0 0" range="-30 30" class="ankle"/>
                <geom name="foot1_{L}" fromto="-.07 -.02 0 .14 -.03 0" size=".027"/>
                <geom name="foot2_{L}" fromto="-.07 .02 0 .14 .03 0" size=".027"/>
              </body>
            </body>
          </body>"""

XML = f"""<mujoco model="hum3d">
  <option timestep="0.004" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".85 .9 .95" rgb2=".5 .6 .7" width="32" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".75 .75 .72" rgb2=".65 .65 .62"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance=".05"/>
    <material name="skin" rgba=".85 .68 .55 1"/>
  </asset>
  <default>
    <geom type="capsule" condim="3" friction="1 .02 .01" material="skin" solimp=".9 .99 .003" solref=".01 1"/>
    <joint type="hinge" damping="1" armature=".02" limited="true"/>
    <default class="big"><joint damping="6" armature=".05"/></default>
    <default class="ankle"><joint damping="4" armature=".05"/></default>
    <default class="arm"><joint damping="1.5"/></default>
    <default class="elbow"><joint damping="1"/></default>
    <default class="wrist"><joint damping=".3" armature=".005"/></default>
    <default class="finger"><joint damping=".05" armature=".001"/></default>
    <position ctrllimited="false"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05" material="grid" condim="3"/>
    <light pos="1 -2 4" dir="-.3 .5 -1" diffuse=".8 .8 .8" castshadow="true"/>
    <light pos="-2 2 3" dir=".5 -.5 -1" diffuse=".3 .3 .3" castshadow="false"/>
    <camera name="front" pos="2.4 -1.2 1.4" xyaxes=".45 .9 0 -.25 .12 .96"/>
    <camera name="side" pos="0.4 -2.6 1.1" xyaxes="1 0 0 0 .1 1"/>
    <body name="torso" pos="0 0 1.282">
      <freejoint name="root"/>
      <geom name="torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="waist_upper" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <site name="torso" pos="0 0 0" size=".01"/>
      <body name="head" pos="0 0 .19">
        <joint name="neck_z" pos="0 0 -.09" axis="0 0 1" range="-70 70" class="arm"/>
        <joint name="neck_y" pos="0 0 -.09" axis="0 1 0" range="-40 50" class="arm"/>
        <geom name="head" type="sphere" size=".09"/>
        <camera name="eyes" pos=".09 0 .02" xyaxes="0 -1 0 .3 0 1" fovy="90"/>
      </body>
      {_arm('left')}{_arm('right')}
      <body name="waist_lower" pos="-.01 0 -.26">
        <joint name="abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="big"/>
        <joint name="abdomen_y" pos="0 0 .065" axis="0 1 0" range="-30 75" class="big"/>
        <geom name="waist_lower" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <body name="pelvis" pos="0 0 -.165">
          <joint name="abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="big"/>
          <geom name="butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          {_leg('left')}{_leg('right')}
        </body>
      </body>
    </body>
    <body name="probe" mocap="true" pos="0 3 1">
      <geom name="probe" type="capsule" fromto="0 0 0 0 0 .12" size=".012" rgba=".2 .4 .9 1" contype="2" conaffinity="1"/>
    </body>
    <body name="ball" pos="0 -3 .5" gravcomp="1e-6">
      <freejoint name="ball"/>
      <geom name="ball" type="sphere" size=".04" rgba=".9 .3 .2 1" mass=".1" condim="4" friction="1.5 .05 .01"/>
    </body>
    <body name="pedestal" mocap="true" pos="0 -3 0">
      <geom name="pedestal" type="capsule" fromto="0 0 -.12 0 0 .12" size=".018" rgba=".35 .55 .35 1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="mount" body1="torso" active="false" solref=".02 1"/>
  </equality>
  <actuator>__ACT__</actuator>
</mujoco>"""

KP = dict(abdomen=400, neck=60, shoulder=250, elbow=150, wrist=30, index=3, middle=3, thumb=3, hip=600, knee=500, ankle=300)

def build():
    xml = XML.replace("__ACT__", "")
    m0 = mujoco.MjModel.from_xml_string(xml); acts = []
    for j in range(m0.njnt):
        n = m0.joint(j).name
        if m0.jnt_type[j] != 3: continue
        kp = KP[n.split("_")[0].rstrip("12")]
        acts.append(f'<position name="{n}" joint="{n}" kp="{kp}" kv="{kp*0.05:.3f}"/>')
    return mujoco.MjModel.from_xml_string(XML.replace("__ACT__", "\n".join(acts)))

M = build()
JN = [M.joint(j).name for j in range(M.njnt) if M.jnt_type[j] == 3]
NJ = len(JN); JQ = np.array([M.jnt_qposadr[M.joint(n).id] for n in JN]); JV = np.array([M.jnt_dofadr[M.joint(n).id] for n in JN])
JR = np.array([M.jnt_range[M.joint(n).id] for n in JN])

# body parts
PARTS = ["head", "chest", "belly", "waist", "hips", "left upper arm", "left forearm", "left hand", "right upper arm", "right forearm",
         "right hand", "left thigh", "left shin", "left foot", "right thigh", "right shin", "right foot"]
def geom_part(n):
    if n in ("head",): return "head"
    if n == "torso": return "chest"
    if n == "waist_upper": return "belly"
    if n == "waist_lower": return "waist"
    if n == "butt": return "hips"
    for side in ("left", "right"):
        if n.endswith(side):
            b = n[:-len(side) - 1]
            if b == "upper_arm": return f"{side} upper arm"
            if b == "forearm": return f"{side} forearm"
            if b in ("palm",) or b[:-1] in ("index", "middle", "thumb"): return f"{side} hand"
            if b == "thigh": return f"{side} thigh"
            if b == "shin": return f"{side} shin"
            if b.startswith("foot"): return f"{side} foot"
    return None
BODY_GEOMS = [g for g in range(M.ngeom) if geom_part(M.geom(g).name)]
GPART = {g: PARTS.index(geom_part(M.geom(g).name)) for g in BODY_GEOMS}

def _surface_points(g, spacing, rng):
    t = M.geom_type[g]; sz = M.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = sz[0]; n = max(8, int(4 * np.pi * r * r / spacing ** 2)); v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * r, v
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = sz[0], sz[1]; area = 2 * np.pi * r * 2 * h + 4 * np.pi * r * r; n = max(6, int(area / spacing ** 2))
        P, N = [], []
        while len(P) < n:
            v = rng.normal(size=3); v /= np.linalg.norm(v); z = rng.uniform(-h - r, h + r)
            if abs(z) <= h: v[2] = 0; v /= np.linalg.norm(v); p = v * r + [0, 0, z]
            else:
                if np.sign(v[2]) != np.sign(z): v[2] = -v[2]
                p = v * r + [0, 0, np.sign(z) * h]
                if rng.uniform() > 1: continue
            P.append(p); N.append(v)
        return np.array(P), np.array(N)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        P, N = [], []
        a = 8 * (sz[0] * sz[1] + sz[1] * sz[2] + sz[0] * sz[2]); n = max(8, int(a / spacing ** 2))
        while len(P) < n:
            ax = rng.integers(3); sg = rng.choice([-1, 1]); p = rng.uniform(-1, 1, 3) * sz; p[ax] = sg * sz[ax]; nn = np.zeros(3); nn[ax] = sg
            P.append(p); N.append(nn)
        return np.array(P), np.array(N)

def _farthest(P, k, rng):
    if len(P) <= k: return np.arange(len(P))
    idx = [rng.integers(len(P))]; d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1): i = int(d.argmax()); idx.append(i); d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(idx)

def build_skin(spacing=0.045, seed=0):
    rng = np.random.default_rng(seed); G, P, N = [], [], []
    for g in BODY_GEOMS:
        sp = spacing * (0.5 if geom_part(M.geom(g).name).endswith("hand") else 1.0)
        p, n = _surface_points(g, sp / 3, rng)  # oversample then even out
        t = M.geom_type[g]; sz = M.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE: area = 2 * np.pi * sz[0] * 2 * sz[1] + 4 * np.pi * sz[0] ** 2
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE: area = 4 * np.pi * sz[0] ** 2
        else: area = 8 * (sz[0] * sz[1] + sz[1] * sz[2] + sz[0] * sz[2])
        k = max(4, int(round(area / sp ** 2))); i = _farthest(p, k, rng)
        G += [g] * len(i); P.append(p[i]); N.append(n[i])
    return np.array(G), np.concatenate(P), np.concatenate(N)

SKIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skin.npz")
if os.path.exists(SKIN_PATH):
    _z = np.load(SKIN_PATH); TG, TP, TN = _z["G"], _z["P"], _z["N"]
else:
    TG, TP, TN = build_skin(); np.savez(SKIN_PATH, G=TG, P=TP, N=TN)
NT = len(TG); TPART = np.array([GPART[g] for g in TG])
TAX_BY_GEOM = {g: np.where(TG == g)[0] for g in BODY_GEOMS}
SIGMA = 0.02

def taxel_world(d):
    """world positions and normals of all taxels"""
    R = d.geom_xmat[TG].reshape(-1, 3, 3); return d.geom_xpos[TG] + np.einsum("nij,nj->ni", R, TP), np.einsum("nij,nj->ni", R, TN)

_f6 = np.zeros(6)
def skin(m, d):
    """taxel pressures from all contacts touching body geoms. returns (act[NT], list of (geom_a, geom_b))"""
    a = np.zeros(NT); pairs = []
    for i in range(d.ncon):
        c = d.contact[i]; mujoco.mj_contactForce(m, d, i, _f6); f = abs(_f6[0])
        if f < 1e-4: continue
        pairs.append((c.geom1, c.geom2, f))
        for g in (c.geom1, c.geom2):
            ix = TAX_BY_GEOM.get(g)
            if ix is None: continue
            R = d.geom_xmat[g].reshape(3, 3); pl = R.T @ (c.pos - d.geom_xpos[g])
            dd = np.sum((TP[ix] - pl) ** 2, 1); a[ix] += f * np.exp(-dd / (2 * SIGMA ** 2))
    return a, pairs
