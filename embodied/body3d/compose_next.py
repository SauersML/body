"""before/after practice videos. python compose_next.py ARM  -> next/<arm>_<task>.mp4
Left: before practice (r0). Right: after 4 rounds (r4). Same 6 scenes, one after another."""
import sys, glob, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
arm = sys.argv[1]; W, H = 960, 720; BAR = 110
F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 64)
ARMLAB = {"sim": "after practice, simulator judges", "self": "after practice, Qwen judges", "ctrl": "after more expert training"}
TASK = {"reach": "Touch your <part>", "poke_reach": "Touch where you were touched", "grasp": "Grasp the handle"}
def read(fn):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", fn, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
def bar(text, w):
    im = Image.new("RGB", (w, BAR), "white"); d = ImageDraw.Draw(im)
    tw = d.textlength(text, font=F); d.text(((w - tw) / 2, 20), text, font=F, fill=(29, 29, 31)); return np.array(im)
for task in ("reach", "poke_reach", "grasp"):
    out = f"next/{arm}_{task}.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2*W}x{H+BAR}", "-r", "25", "-i", "-",
                          "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", out], stdin=subprocess.PIPE)
    hdr = np.concatenate([bar("before", W), bar(ARMLAB[arm], W)], 1)
    for j in range(6):
        fa = glob.glob(f"next/raw/{arm}_r0_{task}_{j}_*.mp4"); fb = glob.glob(f"next/raw/{arm}_r4_{task}_{j}_*.mp4")
        if not fa or not fb: continue
        A, B = read(fa[0]), read(fb[0]); n = max(len(A), len(B)) + 25
        oka, okb = "_ok" in fa[0], "_ok" in fb[0]
        for t in range(n):
            fr = []
            for X, ok in ((A, oka), (B, okb)):
                f = X[min(t, len(X) - 1)].copy()
                if t >= len(X) - 1:  # outcome border once the episode ends
                    c = (40, 160, 70) if ok else (200, 40, 40); f[:12] = c; f[-12:] = c; f[:, :12] = c; f[:, -12:] = c
                fr.append(f)
            p.stdin.write(np.concatenate([hdr, np.concatenate(fr, 1)], 0).tobytes())
    p.stdin.close(); p.wait(); print(out)
