import numpy as np, bodies as BD
def proj(b, xyz):
    x, y, z = xyz
    if b in ("dog", "bird"): return (x - 1.5 * z, y)
    if b in ("fish", "car"): return (x - 0.6 * z, y)
    if b == "spider": return (x, y)
    if b == "snake": return (x, 9 * np.sin(x / 15.0) + 3 * y)
    if b == "octopus": return (x, y + 0.35 * z)
    if b == "drone": return (-z, x * 0.55 + y * 1.4)
    if b == "starfish": return (x, z + 4 * y)
    if b == "jellyfish": return (x + 0.35 * z, y + 0.15 * z)
    if b == "tree": return (x, y)
    if b == "blob": return (x - 0.42 * z, y - 0.3 * z)
