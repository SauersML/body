# Calm drone that brightens as the touch map's match to the skin rises (OLMo-2 stages, same timing as em.js)
import json, sys, os, numpy as np
sys.path.insert(0, os.path.expanduser('~/fig/bodyvid')); from audio import synth
S = json.load(open('web/stages.json')); N = len(S['Y']); FPS = 30; HOLD, MORPH, INTRO, OUTRO = .9, 1.0, 4.0, 4.0
T = INTRO + N * (HOLD + MORPH) - MORPH + OUTRO
rho = np.array(json.load(open('rho.json')))
def val(t):
    u = max(0, t - INTRO); s = int(u // (HOLD + MORPH)); f = (u - s * (HOLD + MORPH) - HOLD) / MORPH
    if s >= N - 1: return rho[-1]
    e = np.clip(f, 0, 1); e = e * e * (3 - 2 * e); return (1 - e) * rho[s] + e * rho[s + 1]
synth(np.array([val(k / FPS) for k in range(int(T * FPS))]), FPS, 'em.wav')
