import os
import subprocess, numpy as np, matplotlib.pyplot as plt
from make import *
W, H = 1920, 1080
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.text(.5, .965, "Qwen3-8B's states for 407 touched body spots lay out a body", ha="center", va="top", fontsize=40, weight="bold")
fig.text(.5, .895, "layer 27, PCA with no body labels  ·  lines join neighbouring joints", ha="center", va="top", fontsize=26, color="#222")
ax = fig.add_axes([.05, .13, .9, .74], projection="3d")
legend(fig, .055, 28, XS=(.03, .2, .33, .47, .62, .755), note_x=.93)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                       os.path.expandvars("$BODY_REPO/figures/out/body_pca3d.mp4")], stdin=subprocess.PIPE)
for i in range(300):
    ax.cla(); draw(ax, 15 + 8 * np.sin(2 * np.pi * i / 300), -90 + 360 * i / 300, fs=26, dots=36, zoom=1.12)
    fig.canvas.draw(); ff.stdin.write(bytes(fig.canvas.buffer_rgba()))
    if i in (45, 120, 200): fig.savefig(f"frame{i}.png")
ff.stdin.close(); ff.wait()
