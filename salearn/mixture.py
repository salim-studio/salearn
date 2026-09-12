"""salearn.mixture — GaussianMixture (fast vectorized EM)."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator
from .utils import check_array


class GaussianMixture(BaseEstimator):
    def __init__(self, n_components=1, covariance_type="full", tol=1e-3,
                 max_iter=100, n_init=1, init_params="kmeans", random_state=None,
                 reg_covar=1e-6):
        self.n_components = n_components; self.covariance_type = covariance_type
        self.tol = tol; self.max_iter = max_iter; self.n_init = n_init
        self.init_params = init_params; self.random_state = random_state
        self.reg_covar = reg_covar

    def _estimate(self, X, resp):
        nk = resp.sum(0) + 1e-12
        mu = (resp.T @ X) / nk[:, None]
        if self.covariance_type == "full":
            cov = np.array([((X - mu[k]).T * resp[:, k]) @ (X - mu[k]) / nk[k] + np.eye(X.shape[1]) * self.reg_covar for k in range(self.n_components)])
        elif self.covariance_type == "diag":
            cov = np.array([(((X - mu[k]) ** 2 * resp[:, k][:, None]).sum(0) / nk[k]) + self.reg_covar for k in range(self.n_components)])
        elif self.covariance_type == "spherical":
            v = np.array([(((X - mu[k]) ** 2).sum(1) * resp[:, k]).sum() / (nk[k] * X.shape[1]) + self.reg_covar for k in range(self.n_components)])
            cov = v
        else:  # tied
            c = sum(((X - mu[k]).T * resp[:, k]) @ (X - mu[k]) for k in range(self.n_components)) / len(X) + np.eye(X.shape[1]) * self.reg_covar
            cov = np.array([c for _ in range(self.n_components)])
        return mu, cov, nk / nk.sum()

    def _log_prob(self, X, mu, cov):
        n, p = X.shape
        out = np.empty((n, self.n_components))
        for k in range(self.n_components):
            d = X - mu[k]
            if self.covariance_type == "full":
                try:
                    L = np.linalg.cholesky(cov[k])
                    sol = np.linalg.solve(L, d.T)
                    q = (sol ** 2).sum(0)
                    logdet = 2 * np.log(np.diag(L)).sum()
                except np.linalg.LinAlgError:
                    q = ((d) ** 2).sum(1) / max(np.trace(cov[k]) / p, 1e-12)
                    logdet = p * np.log(max(np.trace(cov[k]) / p, 1e-12))
            elif self.covariance_type == "diag":
                q = ((d ** 2) / np.maximum(cov[k], 1e-12)).sum(1)
                logdet = np.log(np.maximum(cov[k], 1e-12)).sum()
            elif self.covariance_type == "spherical":
                q = (d ** 2).sum(1) / max(cov[k], 1e-12)
                logdet = p * np.log(max(cov[k], 1e-12))
            else:
                try:
                    L = np.linalg.cholesky(cov[0])
                    sol = np.linalg.solve(L, d.T)
                    q = (sol ** 2).sum(0)
                    logdet = 2 * np.log(np.diag(L)).sum()
                except np.linalg.LinAlgError:
                    q = (d ** 2).sum(1)
                    logdet = 0.0
            out[:, k] = -0.5 * (p * np.log(2 * np.pi) + logdet + q)
        return out

    def fit(self, X, y=None):
        X = check_array(X)
        rng = np.random.RandomState(self.random_state)
        best = None
        for _ in range(max(1, self.n_init)):
            # init via kmeans-lite
            idx = rng.choice(len(X), self.n_components, replace=False)
            mu = X[idx].copy()
            cov = np.array([np.cov(X.T) + np.eye(X.shape[1]) * self.reg_covar for _ in range(self.n_components)])
            w = np.full(self.n_components, 1 / self.n_components)
            ll_prev = -np.inf
            for it in range(int(self.max_iter)):
                lp = self._log_prob(X, mu, cov) + np.log(np.maximum(w, 1e-300))
                lse = lp.max(1, keepdims=True)
                ll = (np.log(np.exp(lp - lse).sum(1)) + lse.ravel()).sum()
                resp = np.exp(lp - lse) / np.exp(lp - lse).sum(1, keepdims=True)
                mu, cov, w = self._estimate(X, resp)
                if abs(ll - ll_prev) < self.tol:
                    break
                ll_prev = ll
            if best is None or ll > best[0]:
                best = (ll, mu, cov, w, it + 1)
        self.lower_bound_ = best[0]
        self.means_, self.covariances_, self.weights_ = best[1], best[2], best[3]
        self.n_iter_ = best[4]
        self.converged_ = True
        # precisions
        if self.covariance_type == "full":
            self.precisions_ = np.array([np.linalg.inv(c) for c in self.covariances_])
            self.precisions_cholesky_ = np.array([np.linalg.cholesky(p) for p in self.precisions_])
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        self._check_fitted()
        X = check_array(X)
        lp = self._log_prob(X, self.means_, self.covariances_) + np.log(np.maximum(self.weights_, 1e-300))
        lse = lp.max(1, keepdims=True)
        e = np.exp(lp - lse)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def score(self, X, y=None):
        X = check_array(X)
        lp = self._log_prob(X, self.means_, self.covariances_) + np.log(np.maximum(self.weights_, 1e-300))
        lse = lp.max(1, keepdims=True)
        return float((np.log(np.exp(lp - lse).sum(1)) + lse.ravel()).mean())

    def bic(self, X):
        n, p = np.asarray(X).shape
        k = self.n_components
        n_params = k * p + k * p * (p + 1) / 2 + k - 1 if self.covariance_type == "full" else k * p * 2 + k - 1
        return -2 * self.score(X) * n + n_params * np.log(n)

    def aic(self, X):
        n, p = np.asarray(X).shape
        k = self.n_components
        n_params = k * p * 2 + k - 1
        return -2 * self.score(X) * n + 2 * n_params

    def sample(self, n_samples=1):
        rng = np.random.RandomState()
        comp = rng.choice(self.n_components, n_samples, p=self.weights_)
        X = np.array([rng.multivariate_normal(self.means_[c], self.covariances_[c] if self.covariance_type == "full" else np.diag(self.covariances_[c] if self.covariance_type == "diag" else np.full(len(self.means_[c]), self.covariances_[c]))) for c in comp])
        return X, comp
