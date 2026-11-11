exec(open("anal.py").read().split("r = lambda")[0])
r = lambda D, B: spearmanr(D[it], B[it])[0]
L = 25; M = typemeans(L); Mc = M - M.mean(0)
nr = np.linalg.norm(Mc, axis=1); o = np.argsort(-nr)
print("centered norms:", [(tk[i], round(nr[i], 1)) for i in o])
v = Mc.var(0); od = np.argsort(-v); print("var share top5 dims", (v[od[:5]].sum() / v.sum()).round(3), od[:5])
Z = Mc / Mc.std(0); print("euc z-scored dims", r(euc(Z), Gsteps), "cos z", r(cos_d(Z), Gsteps))
Mn = Mc / nr[:, None]; print("euc on unit-normed", r(euc(Mn), Gsteps))
# drop top-variance dims
for k in [1, 3, 10]:
    keep = np.setdiff1d(np.arange(M.shape[1]), od[:k]); print("euc drop top", k, r(euc(Mc[:, keep]), Gsteps), "cos", r(cos_d(Mc[:, keep]), Gsteps))
# drop outlier parts by norm
for k in [1, 2, 3, 4]:
    keep = np.setdiff1d(np.arange(len(tk)), o[:k]); ii = np.triu_indices(len(keep), 1)
    E = euc(M[keep]); C = cos_d(M[keep]); B = Gsteps[np.ix_(keep, keep)]
    print("drop", [tk[i] for i in o[:k]], "euc", round(spearmanr(E[ii], B[ii])[0], 3), "cos", round(spearmanr(C[ii], B[ii])[0], 3))
# euclid = sqrt(n_i^2 + n_j^2 - 2 n_i n_j cos): how much of euc rank is just norm sum
ns = (nr[:, None] ** 2 + nr[None] ** 2); print("rho(euc, norm-sum)", spearmanr(euc(M)[it], ns[it])[0], "rho(norm-sum, body)", spearmanr(ns[it], Gsteps[it])[0])
# subsets
arm = [tk.index(t) for t in tk[:12]]; leg = [tk.index(t) for t in tk[12:20]]
for nm, s in [("arm+leg", arm + leg), ("arm", arm), ("leg", leg)]:
    ii = np.triu_indices(len(s), 1); B = Gsteps[np.ix_(s, s)]
    print(nm, "cos(all-centered)", round(spearmanr(cos_d(M)[np.ix_(s, s)][ii], B[ii])[0], 3), "euc", round(spearmanr(euc(M)[np.ix_(s, s)][ii], B[ii])[0], 3))
