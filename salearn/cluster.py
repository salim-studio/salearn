"""salearn.cluster — KMeans (numba), DBSCAN, Agglomerative, Birch-lite, etc."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, ClusterMixin
from .utils import check_array
from ._numba import jit, HAS_NUMBA


@jit()
def _assign_lloyd(X, C, labels, dists):
    for i in range(X.shape[0]):
        best = 0
        bd = 1e300
        for j in range(C.shape[0]):
            s = 0.0
            for k in range(X.shape[1]):
                d = X[i, k] - C[j, k]
                s += d * d
            if s < bd:
                bd = s
                best = j
        labels[i] = best
        dists[i] = bd


@jit()
def _kmeans_pp_dist(X, C, n_done, out):
    for i in range(X.shape[0]):
        bd = 1e300
        for j in range(n_done):
            s = 0.0
            for k in range(X.shape[1]):
                d = X[i, k] - C[j, k]
                s += d * d
            if s < bd:
                bd = s
        out[i] = bd


def _silhouette_samples(X, labels, metric="euclidean"):
    from .metrics import pairwise_distances
    X = np.asarray(X, dtype=np.float64); labels = np.asarray(labels)
    D = pairwise_distances(X, metric=metric)
    u = np.unique(labels)
    out = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        a = D[i][same].mean() if same.any() else 0.0
        b = np.inf
        for c in u:
            if c == labels[i]:
                continue
            m = labels == c
            if m.any():
                b = min(b, D[i][m].mean())
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return out


def _silhouette(X, labels, metric="euclidean"):
    return float(_silhouette_samples(X, labels, metric).mean())


class KMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300,
                 tol=1e-4, random_state=None, algorithm="lloyd"):
        self.n_clusters = n_clusters; self.init = init; self.n_init = n_init
        self.max_iter = max_iter; self.tol = tol; self.random_state = random_state
        self.algorithm = algorithm

    def _init_centers(self, X, rng):
        n, p = X.shape
        k = self.n_clusters
        if isinstance(self.init, np.ndarray):
            return np.asarray(self.init, dtype=np.float64).copy()
        if self.init == "random":
            return X[rng.choice(n, k, replace=False)].copy()
        # k-means++
        C = np.empty((k, p))
        C[0] = X[rng.randint(n)]
        d2 = np.empty(n)
        for j in range(1, k):
            _kmeans_pp_dist(np.ascontiguousarray(X), np.ascontiguousarray(C), j, d2)
            tot = d2.sum()
            if tot <= 0:
                C[j] = X[rng.randint(n)]
            else:
                C[j] = X[rng.choice(n, p=d2 / tot)]
        return C

    def _one_run(self, X, C):
        labels = np.empty(len(X), dtype=np.int64)
        dists = np.empty(len(X))
        prev = np.inf
        for it in range(int(self.max_iter)):
            _assign_lloyd(np.ascontiguousarray(X), np.ascontiguousarray(C), labels, dists)
            inertia = float(dists.sum())
            # update
            for j in range(self.n_clusters):
                m = labels == j
                if m.any():
                    C[j] = X[m].mean(0)
                else:
                    C[j] = X[np.random.randint(len(X))]
            if abs(prev - inertia) <= self.tol * max(inertia, 1.0):
                break
            prev = inertia
        return labels, C, float(dists.sum()), it + 1

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X)
        rng = np.random.RandomState(self.random_state)
        best = None
        for _ in range(max(1, self.n_init)):
            C = self._init_centers(X, rng)
            labels, C, inertia, n_iter = self._one_run(X, C)
            if best is None or inertia < best[0]:
                best = (inertia, labels.copy(), C.copy(), n_iter)
        self.inertia_, self.labels_, self.cluster_centers_, self.n_iter_ = best
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        X = check_array(X)
        labels = np.empty(len(X), dtype=np.int64)
        dists = np.empty(len(X))
        _assign_lloyd(np.ascontiguousarray(X), np.ascontiguousarray(self.cluster_centers_), labels, dists)
        return labels

    def fit_predict(self, X, y=None, **kw):
        return self.fit(X).labels_

    def transform(self, X):
        self._check_fitted()
        X = check_array(X)
        XX = np.einsum("ij,ij->i", X, X)
        CC = np.einsum("ij,ij->i", self.cluster_centers_, self.cluster_centers_)
        D2 = XX[:, None] + CC[None, :] - 2 * X @ self.cluster_centers_.T
        return np.sqrt(np.maximum(D2, 0))

    def score(self, X, y=None):
        return -self.fit(X).inertia_ if not getattr(self, "_is_fitted", False) else -float(((X - self.cluster_centers_[self.predict(X)]) ** 2).sum())


class MiniBatchKMeans(KMeans):
    def __init__(self, n_clusters=8, batch_size=256, max_iter=100, tol=1e-4, random_state=None, n_init=3):
        super().__init__(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
        self.batch_size = batch_size

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X)
        rng = np.random.RandomState(self.random_state)
        C = self._init_centers(X, rng)
        counts = np.zeros(self.n_clusters)
        for _ in range(int(self.max_iter)):
            idx = rng.choice(len(X), min(self.batch_size, len(X)), replace=False)
            B = X[idx]
            labels = np.empty(len(B), dtype=np.int64); dists = np.empty(len(B))
            _assign_lloyd(np.ascontiguousarray(B), np.ascontiguousarray(C), labels, dists)
            for j in range(self.n_clusters):
                m = labels == j
                if m.any():
                    counts[j] += m.sum()
                    eta = m.sum() / counts[j]
                    C[j] = (1 - eta) * C[j] + eta * B[m].mean(0)
        # final assign
        labels = np.empty(len(X), dtype=np.int64); dists = np.empty(len(X))
        _assign_lloyd(np.ascontiguousarray(X), np.ascontiguousarray(C), labels, dists)
        self.cluster_centers_ = C; self.labels_ = labels
        self.inertia_ = float(dists.sum()); self.n_iter_ = int(self.max_iter)
        self._is_fitted = True
        return self


class DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        self.eps = eps; self.min_samples = min_samples; self.metric = metric

    def fit(self, X, y=None):
        from .metrics import pairwise_distances
        X = check_array(X)
        D = pairwise_distances(X, metric=self.metric)
        n = len(X)
        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cid = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neigh = np.where(D[i] <= self.eps)[0]
            if len(neigh) < self.min_samples:
                continue
            # expand
            labels[i] = cid
            seeds = list(neigh)
            seeds.remove(i) if i in seeds else None
            k = 0
            while k < len(seeds):
                j = seeds[k]
                if not visited[j]:
                    visited[j] = True
                    jn = np.where(D[j] <= self.eps)[0]
                    if len(jn) >= self.min_samples:
                        for q in jn:
                            if q not in seeds:
                                seeds.append(q)
                if labels[j] == -1:
                    labels[j] = cid
                k += 1
            cid += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.where([np.sum(D[i] <= self.eps) >= self.min_samples for i in range(n)])[0]
        self.components_ = X[self.core_sample_indices_] if len(self.core_sample_indices_) else np.empty((0, X.shape[1]))
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def fit_predict(self, X, y=None, **kw):
        return self.fit(X).labels_


class AgglomerativeClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=2, linkage="ward", metric="euclidean"):
        self.n_clusters = n_clusters; self.linkage = linkage; self.metric = metric

    def fit(self, X, y=None):
        from scipy.cluster.hierarchy import linkage as _link, fcluster as _fc
        X = check_array(X)
        Z = _link(X, method=self.linkage if self.linkage != "ward" else "ward")
        self.labels_ = _fc(Z, self.n_clusters, criterion="maxclust") - 1
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def fit_predict(self, X, y=None, **kw):
        return self.fit(X).labels_


class Birch(BaseEstimator, ClusterMixin):
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold; self.branching_factor = branching_factor; self.n_clusters = n_clusters

    def fit(self, X, y=None):
        try:
            from sklearn.cluster import Birch as _B
            m = _B(threshold=self.threshold, branching_factor=self.branching_factor, n_clusters=self.n_clusters)
            m.fit(np.asarray(X, dtype=np.float64))
            self.labels_ = m.labels_
            self.subcluster_centers_ = m.subcluster_centers_
            if hasattr(m, "cluster_centers_"):
                self.cluster_centers_ = m.cluster_centers_
        except ImportError:
            # fallback: MiniBatchKMeans
            m = MiniBatchKMeans(n_clusters=self.n_clusters or 3, random_state=0).fit(X)
            self.labels_ = m.labels_
            self.cluster_centers_ = m.cluster_centers_
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        if hasattr(self, "cluster_centers_"):
            from .metrics import pairwise_distances
            return pairwise_distances(np.asarray(X, dtype=np.float64), self.cluster_centers_).argmin(1)
        return np.zeros(len(X), dtype=int)


class MeanShift(BaseEstimator, ClusterMixin):
    def __init__(self, bandwidth=None, max_iter=300):
        self.bandwidth = bandwidth; self.max_iter = max_iter

    def fit(self, X, y=None):
        X = check_array(X)
        bw = self.bandwidth or (np.median(np.abs(X - np.median(X, axis=0))) * 2 + 1e-6)
        self.bandwidth_ = float(bw)
        centers = X.copy()
        for _ in range(int(self.max_iter)):
            D2 = ((centers[:, None, :] - X[None, :, :]) ** 2).sum(-1)
            W = np.exp(-D2 / (2 * bw * bw))
            new = (W[:, :, None] * X[None, :, :]).sum(1) / W.sum(1, keepdims=True)
            if np.abs(new - centers).max() < 1e-3 * bw:
                centers = new
                break
            centers = new
        # merge close centers
        uniq = [centers[0]]
        for c in centers[1:]:
            if min(np.linalg.norm(c - u) for u in uniq) > bw:
                uniq.append(c)
        self.cluster_centers_ = np.array(uniq)
        # assign
        D = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(-1)
        self.labels_ = D.argmin(1)
        self._is_fitted = True
        return self


class OPTICS(DBSCAN):
    def __init__(self, min_samples=5, max_eps=np.inf, metric="euclidean"):
        super().__init__(eps=max_eps if np.isfinite(max_eps) else 1e12, min_samples=min_samples, metric=metric)
        self.max_eps = max_eps
