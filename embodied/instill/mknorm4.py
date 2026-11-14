"""normalization stats for v4 sensors (sim only, runs on CPU)."""
import numpy as np, sys, multiprocessing as mp
import episodes4 as E, sim4 as S
def work(seed):
    rng = np.random.default_rng(seed); return np.concatenate([E.make_episode(rng)["S"] for _ in range(60)])
if __name__ == "__main__":
    with mp.Pool(6) as p: X = np.concatenate(p.map(work, range(12)))
    mu = X.mean(0); sd = X.std(0)
    sd[S.PROP_DIM:S.PROP_DIM + S.SKIN_DIM] = X[:, S.PROP_DIM:S.PROP_DIM + S.SKIN_DIM].std() + 1e-3  # one shared scale for skin
    mu[S.PROP_DIM:S.PROP_DIM + S.SKIN_DIM] = 0
    sd = np.maximum(sd, 0.05); np.savez("v4/norm4.npz", mu=mu, sd=sd); print(X.shape, sd.round(2))
