"""Hand-drawn silhouettes, in each body's projected (u, v) units (v up). Each function returns a list of (kind, args, layer) where layer 'far' is drawn darker, first."""
import numpy as np
def ell(u, v, ru, rv, rot=0, far=False): return ("ell", (u, v, ru, rv, rot), far)
def limb(pts, w, far=False): return ("limb", (pts, w), far)
def blob(pts, far=False): return ("blob", (pts,), far)   # closed smooth curve through pts
def poly(pts, far=False): return ("poly", (pts,), far)
def taper(pts, w0, w1, far=False): return ("taper", (pts, w0, w1), far)

def dog():
    f, n = 6, -6   # far side shifted right, near side left (u = x - 1.5 z)
    return [limb([(66 + f, 40), (67 + f, 22), (66 + f, 5), (71 + f, 4)], 7, True), limb([(20 + f, 46), (24 + f, 26), (17 + f, 6), (22 + f, 4)], 8, True),
            limb([(80 + f, 79), (83 + f, 90)], 6, True),
            ell(45, 51, 31, 14), ell(68, 47, 12, 14), limb([(70, 56), (80, 70)], 15), ell(85, 73, 11, 9), limb([(86, 69), (96, 63)], 10),
            limb([(18, 60), (10, 70), (4, 80)], 5),
            limb([(66 + n + 4, 40), (70 + n, 22), (70 + n, 5), (75 + n, 4)], 8), limb([(24 + n, 46), (28 + n, 26), (22 + n, 6), (27 + n, 4)], 9),
            limb([(78 + n + 4, 80), (75 + n + 4, 90)], 6)]
def bird():
    return [blob([(70, 62), (66, 52), (58, 44), (44, 42), (32, 48), (42, 60), (56, 66)], True),
            limb([(58, 44), (57, 10), (60, 8)], 3.5, True),
            ell(56, 55, 22, 14, -15), ell(80, 78, 10, 9), poly([(88, 80), (98, 74), (88, 70)]),
            poly([(38, 50), (6, 38), (10, 33), (40, 44)]),
            blob([(62, 64), (52, 58), (30, 54), (8, 52), (26, 50), (46, 52), (60, 58)]),
            limb([(52, 42), (51, 10), (48, 8)], 3.5)]
def spider():
    out = [ell(0, -14, 13, 18), ell(0, 13, 10, 11)]
    for s in (1, -1):
        for i, (du, dv) in enumerate([(30, 40), (36, 16), (34, -6), (26, -30)]):
            base = (s * 6, 18 - 4 * i); tip = (s * (du + 10 + 2 * i), dv + (-10 if i < 2 else -14))
            out.append(limb([base, (s * du, dv + 6 if i < 2 else dv + 4), tip], 3.2))
        out.append(limb([(s * 3, 22), (s * 5, 28)], 2.6))
    out.append(limb([(-1.5, 23), (0, 29)], 2.2)); out.append(limb([(1.5, 23), (0, 29)], 2.2)); out.append(limb([(0, -32), (0, -35)], 3))
    return out
def fish():
    return [poly([(68, 40), (62, 32), (74, 36)], True),
            blob([(99, 50), (92, 60), (76, 66), (52, 67), (30, 58), (20, 51), (30, 43), (52, 33), (76, 34), (92, 41)]),
            poly([(24, 50), (2, 66), (8, 50), (2, 34)]), poly([(38, 64), (50, 82), (64, 66)]), poly([(48, 34), (44, 24), (56, 32)]),
            poly([(62, 42), (52, 34), (60, 44)])]
def snake():
    xs = np.linspace(4, 92, 45)
    pts = [(x, 9 * np.sin(x / 15.0)) for x in xs]
    return [taper(pts, 2.5, 8.5), ell(97, 2.5, 7.5, 5.2, 8), limb([(103, 4), (107, 5), (110, 7)], 1), limb([(107, 5), (110, 3)], 1)]
def octopus():
    out = []
    for i in range(1, 9):
        a = 2 * np.pi * (i - 0.5) / 8; x, y, z = 30 * np.sin(a), 6 - 6 * np.cos(a), 30 * np.cos(a)
        u, v = x, y + 0.35 * z; base = (0.25 * u, 14); tip = (1.45 * u, v - 10 - 4 * abs(np.cos(a)))
        far = z < 0
        out.append(taper([base, (u, v), (1.3 * u, v - 6), tip], 2.2 if far else 1.6, 6, far))
    out += [ell(0, 38, 13, 17), ell(0, 22, 12, 9)]
    return out
def car():
    return [ell(84, 10, 9, 9, 0, True), ell(28, 10, 9, 9, 0, True),
            blob([(0, 14), (0, 36), (8, 46), (26, 48), (36, 70), (60, 70), (76, 50), (98, 44), (102, 32), (102, 14)]),
            ell(72, 10, 9, 9), ell(16, 10, 9, 9)]
def drone():
    out = []
    for (u, v) in [(-30, -16), (30, -16)]: out += [limb([(0, 0), (u, v)], 3, True), ell(u, v + 2, 13, 4, 0, True)]
    out += [limb([(-8, -2), (-10, -13), (-14, -14)], 2.4), limb([(8, -2), (10, -13), (14, -14)], 2.4)]
    out += [blob([(-10, -3), (-10, 6), (0, 10), (10, 6), (10, -3), (0, -5)]), limb([(-6, 6), (-6, 10)], 1.5)]
    for (u, v) in [(-30, 16), (30, 16)]: out += [limb([(0, 0), (u, v)], 3), ell(u, v + 2, 13, 4)]
    return out
def starfish():
    pts = []
    for i in range(5):
        a = 2 * np.pi * i / 5; b = a + np.pi / 5
        pts += [(46 * np.sin(a), 46 * np.cos(a)), (13 * np.sin(b), 13 * np.cos(b))]
    return [("star", (pts,), False)]
def jellyfish():
    out = []
    for i, (u, v) in enumerate([(18, 10), (6, 13), (-18, 10), (-6, 7)]):
        top = (u * 1.25, 35); out.append(limb([top, (u, v), (u + 3, -8), (u - 2, -22), (u + 2, -32)], 1.4, i in (1, 3)))
    out += [blob([(-8, 30), (-6, 18), (-3, 6), (0, 0), (4, 8), (7, 20), (8, 30)]),
            blob([(-27, 35), (-24, 44), (-14, 52), (0, 55), (14, 52), (24, 44), (27, 35), (0, 31)])]
    return out
def tree():
    return [limb([(0, 0), (-14, -8), (-30, -14), (-40, -16)], 4), limb([(0, 0), (2, -20), (0, -40), (3, -46)], 4), limb([(0, 0), (16, -8), (26, -10)], 3),
            blob([(-30, 76), (-44, 88), (-36, 100), (-16, 110), (0, 114), (18, 110), (36, 100), (40, 86), (26, 76), (0, 72)]),
            poly([(-8, 0), (-5, 92), (5, 92), (8, 0)]), limb([(2, 44), (30, 50), (48, 54)], 4.5), limb([(0, 62), (-30, 80)], 5), limb([(0, 64), (30, 82)], 5)]
def blob_():
    a = np.linspace(0, 2 * np.pi, 13)[:-1]; r = 31 + 2.5 * np.sin(3 * a + 0.5)
    return [blob([(ri * np.cos(t), ri * np.sin(t) * 0.97) for ri, t in zip(r, a)])]
DRAW = dict(dog=dog, bird=bird, spider=spider, fish=fish, snake=snake, octopus=octopus, car=car, drone=drone, starfish=starfish, jellyfish=jellyfish, tree=tree, blob=blob_)
