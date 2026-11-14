"""render recorded episodes to mp4 with skin pressure shown as red taxel dots. python render3d.py vids.pkl OUTPREFIX"""
import sys, pickle, numpy as np, mujoco, subprocess, os
os.environ.setdefault("MUJOCO_GL", "cgl")
import hum as H, env as E
V = pickle.load(open(sys.argv[1], "rb")); pre = sys.argv[2]; m = H.M; d = mujoco.MjData(m)
r = mujoco.Renderer(m, 720, 960); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.1, 0, 0.95]; cam.distance = 2.6; cam.elevation = -12
for kind, v in V.items():
    for j, (rec, info, ok) in enumerate(zip(v["traj"], v["info"], v["ok"])):
        if not rec: continue
        fn = f"{pre}_{kind}_{j}_{'ok' if ok else 'fail'}.mp4"; frames = []
        cam.azimuth = 150 if kind != "balance" else 120
        for (qpos, mp, mq, sk) in rec:
            d.qpos[:] = qpos; d.mocap_pos[:] = mp; d.mocap_quat[:] = mq; mujoco.mj_forward(m, d)
            r.update_scene(d, cam); P, N = H.taxel_world(d); sc = r.scene
            for i in np.where(sk > 0.05)[0][:300]:
                if sc.ngeom >= sc.maxgeom: break
                g = sc.geoms[sc.ngeom]; a = float(min(1, np.log1p(5 * float(sk[i])) / 2))
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008 + 0.012 * a, 0, 0]), P[i] + N[i] * 0.004, np.eye(3).flatten(), np.array([1, 0.1, 0.1, 0.4 + 0.6 * a], np.float32))
                sc.ngeom += 1
            frames.append(r.render())
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x720", "-r", "25", "-i", "-",
                              "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", fn], stdin=subprocess.PIPE)
        p.stdin.write(np.stack(frames).tobytes()); p.stdin.close(); p.wait(); print(fn, len(frames))
