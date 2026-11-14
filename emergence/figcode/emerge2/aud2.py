# Drone brightening with the touch map's match to the skin (rho), same schedule as em2.js; soft bells at the chat-training break and each beat.
import json, sys, os, numpy as np, soundfile as sf, pyloudnorm as pyln
sys.path.insert(0, os.path.expanduser('~/fig/bodyvid')); from audio import synth, SR
S = json.load(open('web/stages.json')); Q = S['sched']; N = len(S['Y']); FPS = 30; T = Q['T']
rho = np.array(json.load(open('rho.json')))
def val(t):
    s = 0
    while s < N - 1 and t >= Q['start'][s + 1]: s += 1
    if s < N - 1 and t >= Q['mstart'][s]:
        e = np.clip((t - Q['mstart'][s]) / Q['morph'][s], 0, 1); e = e * e * (3 - 2 * e); return (1 - e) * rho[s] + e * rho[s + 1]
    return rho[s]
synth(np.array([val(k / FPS) for k in range(int(T * FPS))]), FPS, 'drone.wav')
y, sr = sf.read('drone.wav'); t = np.arange(len(y)) / sr
def bell(t0, f, g):
    u = t - t0; m = u >= 0; z = np.zeros_like(t); uu = u[m]
    z[m] = sum(a * np.sin(2 * np.pi * f * r * uu) * np.exp(-uu * d) for r, a, d in [(1, 1, 2.2), (2.76, .35, 4), (5.4, .12, 7)]) * (1 - np.exp(-uu * 400))
    return g * z
b = bell(Q['brk'], 392 * 2, .16) + bell(Q['brk'] + .18, 587.3 * 2, .12) + bell(Q['beat1'], 523.3 * 2, .12) + bell(Q['beat2'], 659.3 * 2, .12)
y = y + np.stack([b * .9, b * 1.1], 1)
m = pyln.Meter(sr); y = pyln.normalize.loudness(y, m.integrated_loudness(y), -16.0); pk = np.abs(y).max()
if pk > .89: y *= .89 / pk
print('LUFS', round(m.integrated_loudness(y), 2), 'peak', round(np.abs(y).max(), 3), 'dur', round(len(y) / sr, 2))
sf.write('em2.wav', y.astype(np.float32), sr, subtype='PCM_24')
