"""Press into air analysis. Same regression as ptime_an.py: proj_t ~ p_t + press_t(upcoming) + press_(t-1), steps 1..8, within-episode
centred, standardized; 95% bootstrap CI over episodes. press labels = the wall run's blocked flags (same action tokens in all conditions).
In air/frozen p_t == 0, so the regression is proj_t ~ press_t + press_(t-1). Also raw slopes (projection units) so magnitudes compare."""
import json, numpy as np, sys
R = json.load(open(sys.argv[1])); T = 10
def zc(x): x = np.asarray(x, float); return x - np.nanmean(x, 1, keepdims=True)
def fit(Jz, Pz, BLf, idx, use_p, std=True):
    rows = [(Jz[idx, t], Pz[idx, t], BLf[idx, t], BLf[idx, t - 1]) for t in range(1, T - 1)]
    y, x1, x2, x3 = [np.concatenate(c) for c in zip(*rows)]
    X = np.stack([x1, x2, x3] if use_p else [x2, x3], 1)
    if std: X = (X - X.mean(0)) / X.std(0); y = (y - y.mean()) / y.std()
    return np.linalg.lstsq(np.c_[X, np.ones(len(y))], y, rcond=None)[0][:-1]
def ci(f, n, B=1000):
    rg = np.random.default_rng(0); v = np.array([f(rg.integers(0, n, n)) for _ in range(B)]); return np.percentile(v, [2.5, 97.5], 0)
out = {}
for tag, D in R.items():
    w = D['wall']; n = len(w['p']); BL = np.array([b + [np.nan] for b in w['blocked']], float)
    BLz = BL[:, :T - 1] - BL[:, :T - 1].mean(1, keepdims=True); BLf = np.concatenate([BLz, np.full((n, 1), np.nan)], 1)
    Pw = zc(w['p']); Z = np.zeros((n, T)); res = {}
    for cond in ('wall', 'air', 'frozen'):
        Jz = zc(D[cond]['proj']); up = cond == 'wall'; P_ = Pw if up else Z
        allidx = np.arange(n)
        b = fit(Jz, P_, BLf, allidx, up); lo, hi = ci(lambda i: fit(Jz, P_, BLf, i, up), n)
        braw = fit(Jz, P_, BLf, allidx, up, std=False); rlo, rhi = ci(lambda i: fit(Jz, P_, BLf, i, up, False), n)
        k = -1  # press_(t-1) is last
        res[cond] = dict(beta_prev=round(b[k], 2), ci=[round(lo[k], 2), round(hi[k], 2)], raw_prev=round(braw[k], 2), raw_ci=[round(rlo[k], 2), round(rhi[k], 2)],
                         beta_next=round(b[-2], 2), raw_p=round(braw[0], 2) if up else None)
        if cond != 'wall': res[cond]['contact_steps'] = int(np.sum(np.array(D[cond]['ncon']) > 0))
    # (a) air/frozen restricted to episodes with no contact at any step
    for cond in ('air', 'frozen'):
        ok = np.where(np.array(D[cond]['ncon']).sum(1) == 0)[0]; Jz = zc(D[cond]['proj'])
        b = fit(Jz, Z, BLf, ok, False); lo, hi = ci(lambda i: fit(Jz, Z, BLf, ok[i], False), len(ok))
        res[cond]['nocontact'] = dict(n_ep=len(ok), beta_prev=round(b[-1], 2), ci=[round(lo[-1], 2), round(hi[-1], 2)])
    # (b) wall with pressure as categories (0/.3/.65/1 dummies, within-episode centred): does press_(t-1) survive a nonlinear pressure term?
    Pr = np.array(w['p']); lv = [0.3, 0.65, 1.0]; Jz = zc(w['proj'])
    def fitc(idx, std=True):
        rows = [[Jz[idx, t]] + [zc((np.abs(Pr - l) < .02).astype(float))[idx, t] for l in lv] + [BLf[idx, t], BLf[idx, t - 1]] for t in range(1, T - 1)]
        cols = [np.concatenate(c) for c in zip(*rows)]; y = cols[0]; X = np.stack(cols[1:], 1)
        if std: X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1); y = (y - y.mean()) / y.std()
        return np.linalg.lstsq(np.c_[X, np.ones(len(y))], y, rcond=None)[0][:-1]
    b = fitc(np.arange(n)); lo, hi = ci(lambda i: fitc(i), n); braw = fitc(np.arange(n), False)
    res['wall']['catp'] = dict(beta_prev=round(b[-1], 2), ci=[round(lo[-1], 2), round(hi[-1], 2)], raw_prev=round(braw[-1], 2))
    # (c) marginal correlation of proj_t with press_(t-1), each condition
    for cond in ('wall', 'air', 'frozen'):
        Jz = zc(D[cond]['proj']); x = np.concatenate([BLf[:, t - 1] for t in range(1, T)]); y = np.concatenate([Jz[:, t] for t in range(1, T)])
        res[cond]['r_prev'] = round(float(np.corrcoef(x, y)[0, 1]), 2)
    # (d) same regression plus step fixed effects (two-way demeaning: episode and step), to remove the press-phase / time trend
    def sc(M): M = np.array(M, float); return M - np.nanmean(M, 0, keepdims=True)
    BLf2 = sc(BLf)
    for cond in ('wall', 'air', 'frozen'):
        up = cond == 'wall'; Jz = sc(zc(D[cond]['proj'])); P_ = sc(Pw) if up else Z
        b = fit(Jz, P_, BLf2, np.arange(n), up); lo, hi = ci(lambda i: fit(Jz, P_, BLf2, i, up), n)
        braw = fit(Jz, P_, BLf2, np.arange(n), up, False)
        res[cond]['stepfe'] = dict(beta_prev=round(b[-1], 2), ci=[round(lo[-1], 2), round(hi[-1], 2)], beta_next=round(b[-2], 2), raw_prev=round(braw[-1], 2))
    print('   stepFE', {c: res[c]['stepfe'] for c in ('wall', 'air', 'frozen')})
    res['n_press'] = int(np.nansum(BL)); out[tag] = res
    print('   nocontact air', res['air']['nocontact'], 'frozen', res['frozen']['nocontact'], '| wall catp', res['wall']['catp'], '| r_prev', [res[c]['r_prev'] for c in ('wall', 'air', 'frozen')])
    print(tag, 'presses', res['n_press'], ' | '.join(f"{c}: beta_prev {res[c]['beta_prev']} {res[c]['ci']} raw {res[c]['raw_prev']} {res[c]['raw_ci']} next {res[c]['beta_next']}" for c in ('wall', 'air', 'frozen')),
          'raw_p', res['wall']['raw_p'], 'contacts air/frz', res['air']['contact_steps'], res['frozen']['contact_steps'])
print('\nSUMMARY  beta_prev [95% CI] (raw slope, projection units)')
for k, r in out.items():
    f = lambda d: f"{d['beta_prev']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}] ({d['raw_prev']:+.1f})"
    print(f"{k:11s} same-reg  wall {f(r['wall'])}  air {f(r['air'])}  frozen {f(r['frozen'])}")
    print(f"{'':11s} step-FE   wall {f(r['wall']['stepfe'])}  air {f(r['air']['stepfe'])}  frozen {f(r['frozen']['stepfe'])}")
    print(f"{'':11s} wall cat-p {f(r['wall']['catp']) if 'raw_prev' in r['wall']['catp'] else ''}  r_prev wall/air/frozen {r['wall']['r_prev']}/{r['air']['r_prev']}/{r['frozen']['r_prev']}  air nocontact {r['air']['nocontact']['beta_prev']:+.2f} n={r['air']['nocontact']['n_ep']}")
json.dump(out, open(sys.argv[1].replace('.json', '_an.json'), 'w'), indent=1)
