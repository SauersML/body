import numpy as np, episodes as P, env as E
rng = np.random.default_rng(123); S, A = [], []
for i in range(1200):
    e = P.make_episode(rng); S.append(e["S"]); A.append(e["A"])
S = np.concatenate(S); A = np.concatenate(A)
mu = S.mean(0); sd = np.maximum(S.std(0), 0.05); sl = slice(E.PROP_DIM, E.PROP_DIM + E.SKIN_DIM)
mu[sl] = 0; sd[sl] = max(0.05, float(S[:, sl].std()))  # skin: one global scale, zero = untouched
np.savez("norm.npz", mu=mu, sd=sd, amu=A.mean(0), asd=np.maximum(A.std(0), 0.02))
print("skin sd", sd[sl][0], "frac active", (S[:, sl] > 0.01).mean())
