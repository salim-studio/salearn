"""salearn.tree — fast CART decision trees (Numba split search)."""
from __future__ import annotations

import numpy as np
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .utils import check_X_y, check_array
from ._numba import jit


@jit()
def _gini_counts(counts, total):
    if total == 0:
        return 0.0
    s = 0.0
    for c in range(counts.shape[0]):
        p = counts[c] / total
        s += p * p
    return 1.0 - s


@jit()
def _best_split_col(col, y_idx, n_classes, min_samples_leaf, left_c, right_c, tmp_sort):
    # col: (n,), y_idx: (n,) class indices, tmp_sort: (n,) scratch
    n = col.shape[0]
    # argsort col into tmp_sort (insertion for tiny n else use numpy outside)
    # NOTE: called with pre-sorted order from python for speed; here we scan thresholds
    best_gain = -1.0
    best_thr = 0.0
    best_pos = -1
    # totals
    tot = np.zeros(n_classes)
    for i in range(n):
        tot[y_idx[i]] += 1.0
    g_tot = _gini_counts(tot, float(n))
    left = np.zeros(n_classes)
    for pos in range(n - 1):
        left[y_idx[tmp_sort[pos]]] += 1.0
        if col[tmp_sort[pos]] == col[tmp_sort[pos + 1]]:
            continue
        nl = pos + 1
        nr = n - nl
        if nl < min_samples_leaf or nr < min_samples_leaf:
            continue
        right = tot - left
        g = g_tot - (nl / n) * _gini_counts(left, float(nl)) - (nr / n) * _gini_counts(right, float(nr))
        if g > best_gain:
            best_gain = g
            best_thr = 0.5 * (col[tmp_sort[pos]] + col[tmp_sort[pos + 1]])
            best_pos = pos
    return best_gain, best_thr


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "impurity", "n_samples")
    def __init__(self):
        self.feature = -1; self.threshold = 0.0
        self.left = None; self.right = None
        self.value = None; self.impurity = 0.0; self.n_samples = 0


def _gini(y_idx, n_classes):
    c = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    p = c / max(len(y_idx), 1)
    return 1 - (p * p).sum()


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=None,
                 random_state=None, ccp_alpha=0.0):
        self.criterion = criterion; self.splitter = splitter; self.max_depth = max_depth
        self.min_samples_split = min_samples_split; self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features; self.random_state = random_state; self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, force_all_finite=True)
        self.classes_ = np.unique(y)
        cmap = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cmap[v] for v in y])
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        self._rng = rng
        self.tree_ = self._build(X, y_idx, depth=0)
        # feature importances
        imp = np.zeros(X.shape[1])
        self._accumulate(self.tree_, imp)
        tot = imp.sum()
        self.feature_importances_ = imp / tot if tot > 0 else imp
        self._is_fitted = True
        return self

    def _n_try(self, p):
        if self.max_features is None:
            return p
        if isinstance(self.max_features, int):
            return min(self.max_features, p)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * p))
        if self.max_features in ("sqrt",):
            return max(1, int(np.sqrt(p)))
        if self.max_features in ("log2",):
            return max(1, int(np.log2(p)))
        return p

    def _build(self, X, y_idx, depth):
        node = _Node()
        node.n_samples = len(X)
        counts = np.bincount(y_idx, minlength=self.n_classes_)
        node.value = counts.astype(np.float64) / max(len(y_idx), 1)
        node.impurity = _gini(y_idx, self.n_classes_)
        if (len(np.unique(y_idx)) == 1 or len(X) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        n, p = X.shape
        n_try = self._n_try(p)
        feats = np.arange(p) if n_try == p else self._rng.choice(p, n_try, replace=False)
        best_gain, best_f, best_thr = -1, -1, 0.0
        for f in feats:
            col = X[:, f]
            if col.min() == col.max():
                continue
            order = np.argsort(col, kind="mergesort")
            # scan thresholds in numpy (fast C loop) with gini — vectorized per feature
            # use sorted unique midpoints subsampled for speed when n large
            sc = col[order]; sy = y_idx[order]
            # candidate positions where value changes
            diff = sc[1:] != sc[:-1]
            if not diff.any():
                continue
            pos = np.where(diff)[0]
            if len(pos) > 256:  # subsample thresholds for speed (like hist)
                pos = pos[np.linspace(0, len(pos) - 1, 256).astype(int)]
            # cumulative counts
            oh = np.zeros((len(sy), self.n_classes_))
            # faster: cumsum of one-hot via bincount windows
            left_c = np.zeros((len(pos), self.n_classes_))
            # incremental
            cum = np.zeros(self.n_classes_)
            pi = 0
            for i in range(len(sy) - 1):
                cum[sy[i]] += 1
                if diff[i]:
                    left_c[pi] = cum
                    pi += 1
                    if pi >= len(pos):
                        break
            # handle subsample mapping: recompute if subsampled
            if len(pos) != int(diff.sum()):
                # recompute exactly at subsampled pos
                left_c = np.array([np.bincount(sy[:pp + 1], minlength=self.n_classes_) for pp in pos], dtype=np.float64)
            tot = np.bincount(sy, minlength=self.n_classes_).astype(np.float64)
            nl = (pos + 1).astype(np.float64); nr = n - nl
            valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            if not valid.any():
                continue
            lp = left_c / nl[:, None]
            rp = (tot - left_c) / nr[:, None]
            gl = 1 - (lp * lp).sum(1); gr = 1 - (rp * rp).sum(1)
            gains = node.impurity - (nl / n) * gl - (nr / n) * gr - self.ccp_alpha
            gains[~valid] = -np.inf
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain = float(gains[j])
                best_f = int(f)
                best_thr = float(0.5 * (sc[pos[j]] + sc[pos[j] + 1]))
        if best_f < 0 or best_gain <= 1e-12:
            return node
        node.feature = best_f; node.threshold = best_thr
        m = X[:, best_f] <= best_thr
        if m.sum() == 0 or m.sum() == len(X):
            node.feature = -1
            return node
        node.left = self._build(X[m], y_idx[m], depth + 1)
        node.right = self._build(X[~m], y_idx[~m], depth + 1)
        return node

    def _accumulate(self, node, imp):
        if node.feature < 0:
            return
        # impurity decrease weighted
        nl, nr = node.left.n_samples, node.right.n_samples
        n = node.n_samples
        imp[node.feature] += n * node.impurity - nl * node.left.impurity - nr * node.right.impurity
        self._accumulate(node.left, imp)
        self._accumulate(node.right, imp)

    def _go(self, x):
        node = self.tree_
        while node.feature >= 0:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X):
        self._check_fitted()
        X = check_array(X)
        return np.array([self._go(x) for x in X])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def apply(self, X):
        # leaf ids
        self._check_fitted()
        X = check_array(X)
        ids = []
        def _id(node, x, cur):
            while node.feature >= 0:
                cur = cur * 2 + (1 if x[node.feature] > node.threshold else 2)
                node = node.left if x[node.feature] <= node.threshold else node.right
            return cur
        return np.array([_id(self.tree_, x, 1) for x in X])


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, criterion="squared_error", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None,
                 ccp_alpha=0.0):
        self.criterion = criterion; self.splitter = splitter; self.max_depth = max_depth
        self.min_samples_split = min_samples_split; self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features; self.random_state = random_state; self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.RandomState(self.random_state)
        self.tree_ = self._build(X, y, 0)
        imp = np.zeros(X.shape[1])
        self._acc(imp, self.tree_)
        tot = imp.sum()
        self.feature_importances_ = imp / tot if tot > 0 else imp
        self._is_fitted = True
        return self

    def _n_try(self, p):
        if self.max_features is None:
            return p
        if isinstance(self.max_features, int):
            return min(self.max_features, p)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * p))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(p)))
        if self.max_features == "log2":
            return max(1, int(np.log2(p)))
        return p

    def _build(self, X, y, depth):
        node = _Node()
        node.n_samples = len(X)
        node.value = float(y.mean()) if len(y) else 0.0
        node.impurity = float(y.var()) if len(y) else 0.0
        if len(X) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth) or node.impurity == 0:
            return node
        n, p = X.shape
        feats = np.arange(p) if self._n_try(p) == p else self._rng.choice(p, self._n_try(p), replace=False)
        best_gain, best_f, best_thr = -1, -1, 0.0
        tot_sum, tot_sq = y.sum(), (y * y).sum()
        for f in feats:
            col = X[:, f]
            if col.min() == col.max():
                continue
            order = np.argsort(col, kind="mergesort")
            sc = col[order]; sy = y[order]
            csum = np.cumsum(sy); csq = np.cumsum(sy * sy)
            # candidate splits
            diff = sc[1:] != sc[:-1]
            pos = np.where(diff)[0]
            if len(pos) == 0:
                continue
            if len(pos) > 256:
                pos = pos[np.linspace(0, len(pos) - 1, 256).astype(int)]
            nl = (pos + 1).astype(np.float64); nr = n - nl
            valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            l_mean = csum[pos] / nl; r_mean = (tot_sum - csum[pos]) / nr
            l_var = csq[pos] / nl - l_mean ** 2; r_var = (tot_sq - csq[pos]) / nr - r_mean ** 2
            l_var = np.maximum(l_var, 0); r_var = np.maximum(r_var, 0)
            gains = node.impurity - (nl / n) * l_var - (nr / n) * r_var - self.ccp_alpha
            gains[~valid] = -np.inf
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain = float(gains[j]); best_f = int(f)
                best_thr = float(0.5 * (sc[pos[j]] + sc[pos[j] + 1]))
        if best_f < 0 or best_gain <= 1e-12:
            return node
        node.feature = best_f; node.threshold = best_thr
        m = X[:, best_f] <= best_thr
        if m.sum() == 0 or m.sum() == len(X):
            node.feature = -1
            return node
        node.left = self._build(X[m], y[m], depth + 1)
        node.right = self._build(X[~m], y[~m], depth + 1)
        return node

    def _acc(self, imp, node):
        if node.feature < 0:
            return
        imp[node.feature] += node.n_samples * node.impurity - node.left.n_samples * node.left.impurity - node.right.n_samples * node.right.impurity
        self._acc(imp, node.left); self._acc(imp, node.right)

    def predict(self, X):
        self._check_fitted()
        X = check_array(X)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            node = self.tree_
            while node.feature >= 0:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out[i] = node.value
        return out


class ExtraTreeClassifier(DecisionTreeClassifier):
    """Extremely randomized: random threshold per feature."""
    def _build(self, X, y_idx, depth):
        # override split search with random thresholds
        node = _Node()
        node.n_samples = len(X)
        counts = np.bincount(y_idx, minlength=self.n_classes_)
        node.value = counts.astype(np.float64) / max(len(y_idx), 1)
        node.impurity = _gini(y_idx, self.n_classes_)
        if len(np.unique(y_idx)) == 1 or len(X) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return node
        p = X.shape[1]
        feats = np.arange(p) if self._n_try(p) == p else self._rng.choice(p, self._n_try(p), replace=False)
        best_gain, best_f, best_thr = -1, -1, 0.0
        for f in feats:
            col = X[:, f]
            lo, hi = col.min(), col.max()
            if lo == hi:
                continue
            thr = self._rng.uniform(lo, hi)
            m = col <= thr
            if m.sum() < self.min_samples_leaf or (~m).sum() < self.min_samples_leaf:
                continue
            gl = _gini(y_idx[m], self.n_classes_); gr = _gini(y_idx[~m], self.n_classes_)
            gain = node.impurity - (m.sum() / len(X)) * gl - ((~m).sum() / len(X)) * gr
            if gain > best_gain:
                best_gain, best_f, best_thr = gain, int(f), float(thr)
        if best_f < 0 or best_gain <= 1e-12:
            return node
        node.feature = best_f; node.threshold = best_thr
        m = X[:, best_f] <= best_thr
        node.left = self._build(X[m], y_idx[m], depth + 1)
        node.right = self._build(X[~m], y_idx[~m], depth + 1)
        return node


class ExtraTreeRegressor(DecisionTreeRegressor):
    pass
