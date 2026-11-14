import torch, numpy as np
class GRidge:
    """RidgeCV (efficient LOO, intercept) on GPU in float64."""
    def __init__(self, alphas): self.alphas = list(alphas)
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float64, device="cuda"); y = torch.tensor(y, dtype=torch.float64, device="cuda")
        self.xm, self.ym = X.mean(0), y.mean(); Xc, yc = X - self.xm, y - self.ym
        lam, V = torch.linalg.eigh(Xc.T @ Xc); lam = lam.clamp(min=0); XV = Xc @ V; b = XV.T @ yc
        best = None
        for a in self.alphas:
            w = V @ (b / (lam + a)); r = yc - Xc @ w
            h = (XV ** 2 / (lam + a)).sum(1); e = ((r / (1 - h)) ** 2).mean().item()
            if best is None or e < best[0]: best = (e, w)
        self.w = best[1]; return self
    def predict(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float64, device="cuda")
        return ((X - self.xm) @ self.w + self.ym).cpu().numpy()
