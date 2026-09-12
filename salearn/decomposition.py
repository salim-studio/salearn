"""salearn.decomposition — PCA, SVD, NMF, FactorAnalysis, FastICA, Kernels."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, TransformerMixin
from .utils import check_array


class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, svd_solver="auto", whiten=False, random_state=None):
        self.n_components = n_components; self.svd_solver = svd_solver
        self.whiten = whiten; self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        n, p = Xc.shape
        k = self.n_components or min(n, p)
        if isinstance(k, float):
            # variance ratio
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            var = (S ** 2) / (n - 1)
            cum = np.cumsum(var / var.sum())
            k = int(np.searchsorted(cum, k) + 1)
        if self.svd_solver == "randomized" or (self.svd_solver == "auto" and max(n, p) > 500 and k < min(n, p) // 2):
            try:
                from sklearn.utils.extmath import randomized_svd
                U, S, Vt = randomized_svd(Xc, k, random_state=self.random_state)
            except ImportError:
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                U, S, Vt = U[:, :k], S[:k], Vt[:k]
        else:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            U, S, Vt = U[:, :k], S[:k], Vt[:k]
        self.components_ = Vt
        self.singular_values_ = S
        self.explained_variance_ = (S ** 2) / (n - 1)
        tot = ((Xc ** 2).sum() / (n - 1))
        self.explained_variance_ratio_ = self.explained_variance_ / tot if tot > 0 else np.zeros_like(self.explained_variance_)
        self.singular_values_ = S
        if self.whiten:
            self.components_ = Vt / np.sqrt(self.explained_variance_[:, None] + 1e-12) * np.sqrt(n - 1)
        self.n_components_ = k
        self.n_features_in_ = p
        self.noise_variance_ = float(max(tot - self.explained_variance_.sum() / p, 0.0)) if k < p else 0.0
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        Xc = check_array(X) - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        self._check_fitted()
        X = np.asarray(X)
        if self.whiten:
            # approx inverse
            return X @ (self.components_ * 1.0) + self.mean_
        return X @ self.components_ + self.mean_

    def score(self, X, y=None):
        # log-likelihood approx via reconstruction
        Xr = self.inverse_transform(self.transform(X))
        return float(-((np.asarray(X) - Xr) ** 2).mean())


class TruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, algorithm="randomized", random_state=None):
        self.n_components = n_components; self.algorithm = algorithm; self.random_state = random_state

    def fit(self, X, y=None):
        from scipy.sparse.linalg import svds
        import scipy.sparse as sp
        Xs = X if sp.issparse(X) else np.asarray(X, dtype=np.float64)
        k = self.n_components
        try:
            if sp.issparse(Xs):
                U, S, Vt = svds(Xs.astype(np.float64), k=k, random_state=self.random_state)
                order = np.argsort(-S)
                U, S, Vt = U[:, order], S[order], Vt[order]
            else:
                raise ValueError("dense")
        except Exception:
            if sp.issparse(Xs):
                Xs = Xs.toarray()
            U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
            U, S, Vt = U[:, :k], S[:k], Vt[:k]
        self.components_ = Vt
        self.singular_values_ = S
        self.explained_variance_ = (S ** 2) / max(Xs.shape[0] - 1, 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum() if self.explained_variance_.sum() > 0 else self.explained_variance_
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        import scipy.sparse as sp
        if sp.issparse(X):
            return (X @ self.components_.T)
        return np.asarray(X) @ self.components_.T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X) @ self.components_


class NMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, init="nndsvd", max_iter=200, tol=1e-4, random_state=None, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0):
        self.n_components = n_components; self.init = init; self.max_iter = max_iter
        self.tol = tol; self.random_state = random_state
        self.alpha_W = alpha_W; self.alpha_H = alpha_H; self.l1_ratio = l1_ratio

    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if (X < 0).any():
            raise ValueError("NMF needs non-negative X")
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        k = self.n_components
        W = rng.rand(n, k) + 0.1
        H = rng.rand(k, p) + 0.1
        prev = np.inf
        for _ in range(int(self.max_iter)):
            # multiplicative updates (Lee-Seung)
            H *= (W.T @ X) / np.maximum(W.T @ W @ H, 1e-12)
            W *= (X @ H.T) / np.maximum(W @ H @ H.T, 1e-12)
            err = np.linalg.norm(X - W @ H)
            if abs(prev - err) < self.tol:
                break
            prev = err
        self.components_ = H
        self.reconstruction_err_ = float(np.linalg.norm(X - W @ H))
        self.n_iter_ = self.max_iter
        self._is_fitted = True
        return W

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        self._check_fitted()
        # NNLS per row
        from scipy.optimize import nnls
        X = np.asarray(X, dtype=np.float64)
        W = np.array([nnls(self.components_.T, x)[0] for x in X])
        return W

    def inverse_transform(self, W):
        return np.asarray(W) @ self.components_


class FactorAnalysis(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, max_iter=1000, tol=1e-2, random_state=None):
        self.n_components = n_components; self.max_iter = max_iter; self.tol = tol; self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        n, p = Xc.shape
        k = self.n_components or min(n, p)
        # PCA-init EM
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:k].T * np.sqrt(np.maximum(S[:k] ** 2 / n - 1e-6, 1e-6))
        psi = np.maximum(np.diag(np.cov(Xc.T)) - (W ** 2).sum(1), 1e-6)
        for _ in range(int(self.max_iter)):
            M = W.T @ (W / psi) + np.eye(k)
            Minv = np.linalg.inv(M)
            Wnew = (Xc.T @ (Xc @ (W / psi)) @ Minv / n)
            # E-step stats approx; simplified M-step
            psi_new = np.diag(np.cov((Xc - Xc @ (W / psi) @ Minv @ W.T).T)) + 1e-6
            if np.abs(Wnew - W).max() < self.tol:
                W = Wnew
                psi = np.maximum(psi_new, 1e-6)
                break
            W, psi = Wnew, np.maximum(psi_new, 1e-6)
        self.components_ = W.T
        self.noise_variance_ = psi
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        Xc = check_array(X) - self.mean_
        W = self.components_.T
        M = W.T @ (W / self.noise_variance_) + np.eye(W.shape[1])
        return Xc @ (W / self.noise_variance_) @ np.linalg.inv(M)


class FastICA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components; self.max_iter = max_iter; self.tol = tol; self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        # whiten
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = self.n_components or X.shape[1]
        K = (Vt[:k].T / (S[:k] / np.sqrt(len(X) - 1))) * np.sqrt(len(X) - 1)
        Xw = Xc @ (Vt[:k].T / S[:k] * np.sqrt(len(X) - 1)).T if False else (U[:, :k] * np.sqrt(len(X) - 1))
        rng = np.random.RandomState(self.random_state)
        W = rng.randn(k, k)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        for _ in range(int(self.max_iter)):
            gw = np.tanh(Xw @ W.T)
            gpr = (1 - gw ** 2).mean(0)
            Wnew = gw.T @ Xw / len(X) - gpr[:, None] * W
            # symmetric decorrelation
            U2, _, Vt2 = np.linalg.svd(Wnew)
            Wnew = U2 @ Vt2
            if abs(abs((Wnew * W).sum(1)).min() - 1) < self.tol:
                W = Wnew
                break
            W = Wnew
        self.components_ = W @ (Vt[:k] / S[:k, None] * np.sqrt(len(X) - 1) * 1.0)
        self.mixing_ = np.linalg.pinv(self.components_)
        self._is_fitted = True
        return Xw @ W.T

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        self._check_fitted()
        return (check_array(X) - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        return np.asarray(X) @ self.mixing_.T + self.mean_


class KernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, kernel="rbf", gamma=None, degree=3, coef0=1.0):
        self.n_components = n_components; self.kernel = kernel
        self.gamma = gamma; self.degree = degree; self.coef0 = coef0

    def _K(self, X, Y):
        if self.kernel == "rbf":
            g = self.gamma or 1.0 / X.shape[1]
            XX = (X ** 2).sum(1)[:, None]; YY = (Y ** 2).sum(1)[None, :]
            return np.exp(-g * np.maximum(XX + YY - 2 * X @ Y.T, 0))
        if self.kernel == "linear":
            return X @ Y.T
        if self.kernel == "poly":
            return ( (self.gamma or 1.0) * X @ Y.T + self.coef0) ** self.degree
        if self.kernel == "sigmoid":
            return np.tanh((self.gamma or 1.0) * X @ Y.T + self.coef0)
        raise ValueError("unknown kernel")

    def fit(self, X, y=None):
        X = check_array(X)
        self._X = X
        K = self._K(X, X)
        N = len(X)
        self._K_row_mean = K.mean(0)
        self._K_mean = K.mean()
        Kc = K - K.mean(0)[None, :] - K.mean(1)[:, None] + K.mean()
        w, V = np.linalg.eigh(Kc)
        order = np.argsort(-w)
        w, V = w[order], V[:, order]
        k = self.n_components
        self.eigenvalues_ = np.maximum(w[:k], 0)
        self.eigenvectors_ = V[:, :k] / np.sqrt(np.maximum(self.eigenvalues_, 1e-12))
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_fitted()
        X = check_array(X)
        K = self._K(X, self._X)
        Kc = K - K.mean(1)[:, None] - self._K_row_mean[None, :] + self._K_mean
        return Kc @ self.eigenvectors_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
