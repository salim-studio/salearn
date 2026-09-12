"""salearn.neighbors — KNN with Numba brute-force + vectorized paths."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .utils import check_X_y, check_array
from ._numba import jit


@jit()
def _brute_sq(X, Y, out):
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            s = 0.0
            for k in range(X.shape[1]):
                d = X[i, k] - Y[j, k]
                s += d * d
            out[i, j] = s


class _KNNBase(BaseEstimator):
    def __init__(self, n_neighbors=5, weights="uniform", algorithm="auto",
                 leaf_size=30, p=2, metric="minkowski", n_jobs=None):
        self.n_neighbors = n_neighbors; self.weights = weights
        self.algorithm = algorithm; self.leaf_size = leaf_size
        self.p = p; self.metric = metric; self.n_jobs = n_jobs

    def _dist(self, X, Y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        if self.metric in ("euclidean", "minkowski") and self.p == 2:
            # BLAS path
            XX = np.einsum("ij,ij->i", X, X)
            YY = np.einsum("ij,ij->i", Y, Y)
            D2 = XX[:, None] + YY[None, :] - 2 * X @ Y.T
            np.maximum(D2, 0, out=D2)
            return D2
        if self.metric in ("manhattan",) or (self.metric == "minkowski" and self.p == 1):
            return np.abs(X[:, None, :] - Y[None, :, :]).sum(-1)
        # generic minkowski
        return (np.abs(X[:, None, :] - Y[None, :, :]) ** self.p).sum(-1) ** (1 / self.p)

    def _kneighbors(self, X):
        self._check_fitted()
        X = check_array(X)
        D = self._dist(X, self._fit_X)
        k = min(self.n_neighbors, len(self._fit_X))
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        # sort within k
        row = np.take_along_axis(D, part, axis=1)
        order = np.argsort(row, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        dist = np.take_along_axis(D, idx, axis=1)
        if self.metric in ("euclidean", "minkowski") and self.p == 2:
            np.sqrt(dist, out=dist)
        return dist, idx

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if X is None:
            X = self._fit_X
        old = self.n_neighbors
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        try:
            d, idx = self._kneighbors(X)
        finally:
            self.n_neighbors = old
        if return_distance:
            return d, idx
        return idx


class KNeighborsClassifier(_KNNBase, ClassifierMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=True)
        self._fit_X = X; self._fit_y = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        d, idx = self._kneighbors(X)
        n = len(X)
        P = np.zeros((n, len(self.classes_)))
        cmap = {c: i for i, c in enumerate(self.classes_)}
        if self.weights == "uniform":
            for i in range(n):
                for j in idx[i]:
                    P[i, cmap[self._fit_y[j]]] += 1
            P /= P.sum(1, keepdims=True)
        else:  # distance
            w = 1.0 / np.maximum(d, 1e-12)
            for i in range(n):
                for jj, j in enumerate(idx[i]):
                    P[i, cmap[self._fit_y[j]]] += w[i, jj]
            P /= P.sum(1, keepdims=True)
        return P

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class KNeighborsRegressor(_KNNBase, RegressorMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._fit_X = X; self._fit_y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        d, idx = self._kneighbors(X)
        V = self._fit_y[idx]
        if self.weights == "uniform":
            return V.mean(1) if V.ndim == 2 else V.mean(1)
        w = 1.0 / np.maximum(d, 1e-12)
        if V.ndim == 2:
            return (V * w).sum(1) / w.sum(1)
        return (V * w).sum(1) / w.sum(1)


class NearestNeighbors(_KNNBase):
    def fit(self, X, y=None):
        self._fit_X = check_array(X)
        self._fit_y = None
        self.n_features_in_ = self._fit_X.shape[1]
        self._is_fitted = True
        return self

    def kneighbors_graph(self, X=None, n_neighbors=None, mode="connectivity"):
        import scipy.sparse as sp
        d, idx = self.kneighbors(X, n_neighbors=n_neighbors)
        n = d.shape[0]
        m = self._fit_X.shape[0] if X is None else n
        # graph from X rows to fit rows
        rows = np.repeat(np.arange(n), d.shape[1])
        cols = idx.ravel()
        data = np.ones_like(cols, dtype=np.float64) if mode == "connectivity" else d.ravel()
        n_cols = self._fit_X.shape[0]
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n_cols))

    def radius_neighbors(self, X=None, radius=1.0, return_distance=True):
        X = check_array(self._fit_X if X is None else X)
        D = self._dist(X, self._fit_X)
        dists = [np.sort(D[i][D[i] <= radius]) for i in range(len(X))]
        idx = [np.argsort(D[i][D[i] <= radius]) for i in range(len(X))]
        # fix indices: need original indices
        out_d, out_i = [], []
        for i in range(len(X)):
            m = D[i] <= radius
            ii = np.where(m)[0][np.argsort(D[i][m])]
            out_i.append(ii)
            out_d.append(D[i][ii])
        if return_distance:
            return np.array(out_d, dtype=object), np.array(out_i, dtype=object)
        return np.array(out_i, dtype=object)
