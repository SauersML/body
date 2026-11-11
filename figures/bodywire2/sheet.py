from render import *
fig, axs = plt.subplots(3, 4, figsize=(24, 16), dpi=60)
for ax, (el, az) in zip(axs.flat, [(e, a) for e in (10, 22, 32) for a in (-150, -30, 30, 150)]):
    R = draw(ax, el, az, lw=1.2, dot=4); ax.set_title(f"{el},{az}", fontsize=20)
fig.savefig("sheet.png", facecolor="white")
