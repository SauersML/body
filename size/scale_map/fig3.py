import json; from svg import *
A = "$BODY_RUNS"
M = {"0.6B": (json.load(open(A + "q06c-named/analysis_Qwen3-0.6B_named.json")), 28), "8B": (json.load(open(A + "q8b-named/analysis_Qwen3-8B_named.json")), 36)}
json.dump({m: {k: a[k] for k in ("xyz_r2", "lr_decode", "pose_r2")} for m, (a, n) in M.items()}, open("q3_numbers.json", "w"), indent=1)
F = Fig(2300, 1450)
F.t(60, 160, "joint angles peak half-way", 58, 700); F.t(60, 90, "Qwen3 body controller: where the poke was builds up with depth;", 58, 700)
def panel(x0, y0, w, h, key, title, base=0.0):
    F.t(x0, y0 - 60, title, FS, 700)
    X = lambda f: x0 + f * w; Y = lambda v: y0 + h - h * (max(v, 0) - base) / (1 - base)
    F.line(x0, y0 + h, x0 + w, y0 + h); F.line(x0, y0, x0, y0 + h)
    for v in (0, .5, 1):
        if v >= base: F.t(x0 - 20, Y(v) + 16, f"{round(v*100)}%", FS, 400, INK, "end")
    F.t(x0, y0 + h + 70, "input", FS, 400, INK, "start"); F.t(x0 + w, y0 + h + 70, "output", FS, 400, INK, "end"); F.t(x0 + w / 2, y0 + h + 70, "layer depth", FS, 400, INK, "middle")
    for m, (a, n) in M.items():
        pts = sorted((int(l) / n, v) for l, v in a[key].items()); c = C[m]
        F.raw('<polyline fill="none" stroke="%s" stroke-width="9" points="%s"/>' % (c, " ".join(f"{X(f):.0f},{Y(v):.0f}" for f, v in pts)))
        for f, v in pts: F.raw(f'<circle cx="{X(f):.0f}" cy="{Y(v):.0f}" r="14" fill="{c}"/>')
        if key == "xyz_r2": f, v = pts[-1]; F.t(X(f) + 26, Y(v) + 16 + (34 if m == "0.6B" else -10), "Qwen3-" + m, FS, 700, c)
        else: f, v = pts[2]; F.t(X(f) + 120, Y(v) + (-40 if m == "0.6B" else 190), "Qwen3-" + m, FS, 700, c, "middle")
    return X, Y
X, Y = panel(200, 400, 720, 760, "xyz_r2", "3D spot of the poke, % explained")
a8 = M["8B"][0]["xyz_r2"]["18"]; xs = X(18 / 36)
F.line(xs, Y(a8) + 30, xs, 400 + 760, "#D2432F", 6, "14,10"); F.t(xs + 20, 400 + 620, "adding the poke", FS, 700, "#D2432F"); F.t(xs + 20, 400 + 672, "direction here", FS, 700, "#D2432F"); F.t(xs + 20, 400 + 724, "moves the hand", FS, 700, "#D2432F")
panel(1400, 400, 720, 760, "pose_r2", "41 joint angles, % explained")
F.save("../body_scale_layers_v1")
