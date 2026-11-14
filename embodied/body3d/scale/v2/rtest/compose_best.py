"""Short video of the best policy: real sim renders of held-out episodes, the real prompt as a chat bubble, narration + quiet music.
python compose_best.py VIDS.pkl OUT.mp4 MODELNAME  task:idx ...   (VIDS keys like r4_reach)"""
import sys, pickle, subprocess, os, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "cgl")
from PIL import Image, ImageDraw, ImageFont
import hum as H, env as E
V = pickle.load(open(sys.argv[1], "rb")); OUT = sys.argv[2]; MODEL = sys.argv[3]; SHOTS = [a.split(":") for a in sys.argv[4:]]
W, Hh, TOP = 1080, 1080, 230
m = H.M; d = mujoco.MjData(m); r = mujoco.Renderer(m, Hh - TOP, W)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = [0.05, 0, 0.88]; cam.distance = 2.45; cam.elevation = -10
FB = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 76); FS = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54)
def prompt(kind, info):
    if kind == "reach": return f"Touch your {H.PARTS[info['part']]}."
    return "Touch where you were touched." if kind == "poke_reach" else "Grab the handle."
def bubble(text, sub):
    im = Image.new("RGB", (W, TOP), "white"); g = ImageDraw.Draw(im)
    fb = FB; sz = 76
    while g.textlength(text, font=fb) > W - 140: sz -= 2; fb = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", sz)
    tw = g.textlength(text, font=fb); x0 = (W - tw) / 2 - 40
    g.rounded_rectangle([x0, 40, x0 + tw + 80, 150], 50, fill=(43, 120, 246)); g.text((x0 + 40, 95), text, font=fb, fill="white", anchor="lm")
    sw = g.textlength(sub, font=FS); g.text(((W - sw) / 2, 160), sub, font=FS, fill=(29, 29, 31)); return np.array(im)
def frames(rec, kind):
    out = []; cam.azimuth = 150
    pr = [x[1][E.PROBE_M] for x in rec if abs(x[1][E.PROBE_M][1]) < 2]
    if pr and pr[0][0] < 0: cam.azimuth = 330  # poke from behind: view from behind so the probe is visible
    for (qpos, mp, mq, sk) in rec:
        d.qpos[:] = qpos; d.mocap_pos[:] = mp; d.mocap_quat[:] = mq; mujoco.mj_forward(m, d)
        r.update_scene(d, cam); P, N = H.taxel_world(d); sc = r.scene
        for i in np.where(sk > 0.05)[0][:300]:
            if sc.ngeom >= sc.maxgeom: break
            gg = sc.geoms[sc.ngeom]; a = float(min(1, np.log1p(5 * float(sk[i])) / 2))
            mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008 + 0.012 * a, 0, 0]), P[i] + N[i] * 0.004, np.eye(3).flatten(), np.array([1, 0.1, 0.1, 0.4 + 0.6 * a], np.float32))
            sc.ngeom += 1
        pp = mp[E.PROBE_M]
        if abs(pp[1]) < 2 and sc.ngeom < sc.maxgeom:  # mark the poking probe (blue halo) so the poke is visible at this size
            gg = sc.geoms[sc.ngeom]; mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]), pp, np.eye(3).flatten(), np.array([0.17, 0.37, 0.66, 0.45], np.float32)); sc.ngeom += 1
        out.append(r.render())
    return out
FPS = 25; allf = []; lines = []
for n, (kind, j) in enumerate(SHOTS):
    v = V[f"r4_{kind}"]; j = int(j); info = v["info"][j]; ok = v["ok"][j]
    fr = frames(v["traj"][j], kind); fr += [fr[-1]] * 30
    sub = "red dots: pressure on the skin" if n == 0 else ("first, a blue probe pokes the body" if kind == "poke_reach" else " ")
    top = bubble(prompt(kind, info), sub)
    for t, f in enumerate(fr):
        f = f.copy()
        if t >= len(fr) - 30:
            c = (40, 160, 70) if ok else (200, 40, 40); f[:14] = c; f[-14:] = c; f[:, :14] = c; f[:, -14:] = c
        allf.append(np.concatenate([top, f], 0))
    lines.append((len(allf) - len(fr), prompt(kind, info)))
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(FPS), "-i", "-",
                      "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "19", "/tmp/cb_v.mp4"], stdin=subprocess.PIPE)
p.stdin.write(np.stack(allf).tobytes()); p.stdin.close(); p.wait()
# narration: intro line on shot 1, then each prompt read aloud at its shot start
segs = [(0, f"{MODEL} moves this simulated body. Never seen scenes.")] + [(s / FPS + (3.2 if i == 0 else 0.2), t) for i, (s, t) in enumerate(lines)]
inputs, filt = [], []
for i, (t0, text) in enumerate(segs):
    fn = f"/tmp/cb_n{i}.aiff"; subprocess.run(["say", "-v", "Samantha", "-r", "185", "-o", fn, text]); inputs += ["-i", fn]
    filt.append(f"[{i+1}:a]adelay={int(t0*1000)}|{int(t0*1000)},aformat=channel_layouts=stereo[n{i}]")
dur = len(allf) / FPS; music = os.path.expanduser("~/music/attendance.wav")
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/cb_v.mp4"] + inputs + ["-stream_loop", "-1", "-i", music]
k = len(segs) + 1
filt.append(f"[{k}:a]atrim=0:{dur},volume=0.18,afade=t=out:st={dur-1.5}:d=1.5,aformat=channel_layouts=stereo[mu]")
filt.append("".join(f"[n{i}]" for i in range(len(segs))) + f"[mu]amix=inputs={len(segs)+1}:normalize=0[a]")
subprocess.run(cmd + ["-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-t", str(dur), OUT], check=True)
print(OUT, round(dur, 1), "s")
