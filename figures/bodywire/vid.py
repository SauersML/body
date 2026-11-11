import os
import subprocess
from wire import *
W, H = 1920, 1080
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.text(.5, .97, "Qwen3-8B's body map: skin neighbours joined by lines", ha="center", va="top", fontsize=44, weight="bold")
fig.text(.5, .885, "layer 27, PCA with no body labels  ·  body layout kept, skin grid crumpled inside each part", ha="center", va="top", fontsize=26, color="#222")
ax = fig.add_axes([.05, .12, .9, .76], projection="3d")
legend(fig, .05, 28, XS=(.03, .2, .33, .47, .62, .755), note_x=.93)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                       os.path.expandvars("$BODY_REPO/figures/out/body_wire.mp4")], stdin=subprocess.PIPE)
for i in range(300):
    ax.cla(); wire(ax, 15 + 8 * np.sin(2 * np.pi * i / 300), -90 + 360 * i / 300, fs=24, zoom=1.12, lw=1.3, al=.5, dots=8)
    fig.canvas.draw(); ff.stdin.write(bytes(fig.canvas.buffer_rgba()))
    if i in (0, 60): fig.savefig(f"frame{i}.png")
ff.stdin.close(); ff.wait()
