"""Compose the best-policy video: rendered episodes (render3d.py output) per task, big task label on top, outcome border at the end.
python compose.py RAWPREFIX OUT.mp4 STAGE   (RAWPREFIX_STAGE_<task>_<j>_{ok,fail}.mp4)"""
import sys, glob, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
pre, out, stage = sys.argv[1], sys.argv[2], sys.argv[3]; W, H = 960, 720; BAR = 130
F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 72)
TASK = {"reach": "Touch your {}", "poke_reach": "Touch where you were touched", "grasp": "Grab the handle"}
PARTS = ["head", "chest", "belly", "waist", "hips", "left upper arm", "left forearm", "left hand", "right upper arm", "right forearm",
         "right hand", "left thigh", "left shin", "left foot", "right thigh", "right shin", "right foot"]
import pickle
INFO = pickle.load(open(sys.argv[4], "rb")) if len(sys.argv) > 4 else None


def read(fn):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", fn, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def bar(text, w):
    im = Image.new("RGB", (w, BAR), "white"); d = ImageDraw.Draw(im); f = F
    while d.textlength(text, font=f) > w - 40: f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", f.size - 4)
    d.text(((w - d.textlength(text, font=f)) / 2, (BAR - f.size) / 2 - 4), text, font=f, fill=(29, 29, 31)); return np.array(im)


p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W}x{H + BAR}", "-r", "25", "-i", "-",
                      "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", out], stdin=subprocess.PIPE)
for task in ("reach", "poke_reach", "grasp"):
    fs = sorted(glob.glob(f"{pre}_{stage}_{task}_*.mp4"))[:int(sys.argv[5]) if len(sys.argv) > 5 else 6]
    for k in range(0, len(fs) - 1, 2):
        pair = fs[k:k + 2]; clips = [read(f) for f in pair]; n = max(len(c) for c in clips) + 30
        labs = []
        for f in pair:
            j = int(f.split("_")[-2]); lab = TASK[task]
            if task == "reach" and INFO is not None: lab = lab.format(PARTS[INFO[f"{stage}_{task}"]["info"][j]["part"]])
            labs.append(lab)
        hdr = np.concatenate([bar(l, W) for l in labs], 1)
        for t in range(n):
            fr = []
            for X, f in zip(clips, pair):
                im = X[min(t, len(X) - 1)].copy()
                if t >= len(X) - 1:
                    c = (40, 160, 70) if "_ok" in f else (200, 40, 40); im[:14] = c; im[-14:] = c; im[:, :14] = c; im[:, -14:] = c
                fr.append(im)
            p.stdin.write(np.concatenate([hdr, np.concatenate(fr, 1)], 0).tobytes())
p.stdin.close(); p.wait(); print(out)
