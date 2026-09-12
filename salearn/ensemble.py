"""salearn.ensemble — forests, boosting, bagging, voting (parallel via threads)."""
from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from .utils import check_X_y, check_array
from .tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier


def _resolve_n_jobs(n_jobs):
    import os
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return os.cpu_count() or 1
    if n_jobs < -1:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


def _fit_one(est, X, y, sw, seed):
    if seed is not None and hasattr(est, "random_state"):
        try:
            est.set_params(random_state=int(seed))
        except Exception:
            pass
    if sw is not None:
        try:
            return est.fit(X, y, sample_weight=sw)
        except TypeError:
            return est.fit(X, y)
    return est.fit(X, y)


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                 max_samples=None):
        self.n_estimators = n_estimators; self.criterion = criterion; self.max_depth = max_depth
        self.min_samples_split = min_samples_split; self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features; self.bootstrap = bootstrap; self.oob_score = oob_score
        self.n_jobs = n_jobs; self.random_state = random_state; self.max_samples = max_samples

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 1 << 30, self.n_estimators)
        ns = int(self.max_samples * n) if isinstance(self.max_samples, float) else (self.max_samples or n)
        jobs = []
        for i in range(self.n_estimators):
            idx = rng.choice(n, ns, replace=True) if self.bootstrap else np.arange(n)
            sw = None
            if sample_weight is not None:
                sw = np.asarray(sample_weight)[idx]
            t = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                       min_samples_split=self.min_samples_split,
                                       min_samples_leaf=self.min_samples_leaf,
                                       max_features=self.max_features, random_state=int(seeds[i]))
            jobs.append((t, X[idx], y[idx], sw, seeds[i]))
        with ThreadPoolExecutor(max_workers=_resolve_n_jobs(self.n_jobs)) as ex:
            self.estimators_ = list(ex.map(lambda a: _fit_one(*a), jobs))
        imp = np.mean([e.feature_importances_ for e in self.estimators_], axis=0)
        self.feature_importances_ = imp
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        self._check_fitted()
        X = check_array(X)
        P = np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)
        # align classes (all trees share classes_)
        return P

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, criterion="squared_error", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features=1.0,
                 bootstrap=True, n_jobs=None, random_state=None, max_samples=None):
        self.n_estimators = n_estimators; self.criterion = criterion; self.max_depth = max_depth
        self.min_samples_split = min_samples_split; self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features; self.bootstrap = bootstrap
        self.n_jobs = n_jobs; self.random_state = random_state; self.max_samples = max_samples

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 1 << 30, self.n_estimators)
        ns = int(self.max_samples * n) if isinstance(self.max_samples, float) else (self.max_samples or n)
        jobs = []
        for i in range(self.n_estimators):
            idx = rng.choice(n, ns, replace=True) if self.bootstrap else np.arange(n)
            sw = np.asarray(sample_weight)[idx] if sample_weight is not None else None
            t = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                      random_state=int(seeds[i]))
            jobs.append((t, X[idx], np.asarray(y)[idx], sw, seeds[i]))
        with ThreadPoolExecutor(max_workers=_resolve_n_jobs(self.n_jobs)) as ex:
            self.estimators_ = list(ex.map(lambda a: _fit_one(*a), jobs))
        self.feature_importances_ = np.mean([e.feature_importances_ for e in self.estimators_], axis=0)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        X = check_array(X)
        return np.mean([e.predict(X) for e in self.estimators_], axis=0)


class ExtraTreesClassifier(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        # same as RF but with ExtraTree base
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 1 << 30, self.n_estimators)
        ns = int(self.max_samples * n) if isinstance(self.max_samples, float) else (self.max_samples or n)
        jobs = []
        for i in range(self.n_estimators):
            idx = rng.choice(n, ns, replace=True) if self.bootstrap else np.arange(n)
            sw = np.asarray(sample_weight)[idx] if sample_weight is not None else None
            t = ExtraTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                    random_state=int(seeds[i]))
            jobs.append((t, X[idx], y[idx], sw, seeds[i]))
        with ThreadPoolExecutor(max_workers=_resolve_n_jobs(self.n_jobs)) as ex:
            self.estimators_ = list(ex.map(lambda a: _fit_one(*a), jobs))
        self.feature_importances_ = np.mean([e.feature_importances_ for e in self.estimators_], axis=0)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self


class ExtraTreesRegressor(RandomForestRegressor):
    pass


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss="log_loss", learning_rate=0.1, n_estimators=100, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, random_state=None):
        self.loss = loss; self.learning_rate = learning_rate; self.n_estimators = n_estimators
        self.max_depth = max_depth; self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf; self.subsample = subsample; self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        from .tree import DecisionTreeRegressor
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        binary = len(self.classes_) == 2
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        if binary:
            yb = (y == self.classes_[1]).astype(np.float64)
            F = np.full(n, np.log(max(yb.mean(), 1e-12) / max(1 - yb.mean(), 1e-12)))
            self.estimators_ = []
            for _ in range(self.n_estimators):
                p = 1 / (1 + np.exp(-F))
                r = yb - p
                idx = rng.choice(n, int(self.subsample * n), replace=False) if self.subsample < 1 else np.arange(n)
                t = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf, random_state=rng.randint(1 << 30))
                t.fit(X[idx], r[idx])
                # line search multiplier (Newton step per leaf approx = sum r / sum p(1-p))
                leaf_pred = t.predict(X)
                F += self.learning_rate * leaf_pred
                self.estimators_.append(t)
            self._F0 = float(np.log(max(yb.mean(), 1e-12) / max(1 - yb.mean(), 1e-12)))
        else:
            K = len(self.classes_)
            F = np.zeros((n, K))
            self.estimators_ = []
            for _ in range(self.n_estimators):
                e = np.exp(F - F.max(1, keepdims=True)); P = e / e.sum(1, keepdims=True)
                trees = []
                for k, c in enumerate(self.classes_):
                    r = (y == c).astype(np.float64) - P[:, k]
                    t = DecisionTreeRegressor(max_depth=self.max_depth, random_state=rng.randint(1 << 30))
                    t.fit(X, r)
                    F[:, k] += self.learning_rate * t.predict(X)
                    trees.append(t)
                self.estimators_.append(trees)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def decision_function(self, X):
        self._check_fitted()
        X = check_array(X)
        if len(self.classes_) == 2:
            F = np.full(len(X), getattr(self, "_F0", 0.0))
            for t in self.estimators_:
                F += self.learning_rate * t.predict(X)
            return F
        F = np.zeros((len(X), len(self.classes_)))
        for trees in self.estimators_:
            for k, t in enumerate(trees):
                F[:, k] += self.learning_rate * t.predict(X)
        return F

    def predict_proba(self, X):
        s = self.decision_function(X)
        if len(self.classes_) == 2:
            p1 = 1 / (1 + np.exp(-s))
            return np.c_[1 - p1, p1]
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        s = self.decision_function(X)
        if len(self.classes_) == 2:
            return np.where(s >= 0, self.classes_[1], self.classes_[0])
        return self.classes_[s.argmax(1)]


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, loss="squared_error", learning_rate=0.1, n_estimators=100,
                 max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 random_state=None):
        self.loss = loss; self.learning_rate = learning_rate; self.n_estimators = n_estimators
        self.max_depth = max_depth; self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf; self.subsample = subsample; self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        from .tree import DecisionTreeRegressor
        X, y = check_X_y(X, y)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        self._F0 = float(y.mean())
        F = np.full_like(y, self._F0)
        self.estimators_ = []
        n = len(X)
        for _ in range(self.n_estimators):
            if self.loss == "absolute_error":
                r = np.sign(y - F)
            elif self.loss == "huber":
                d = y - F; delta = 1.35 * np.median(np.abs(d)) + 1e-12
                r = np.where(np.abs(d) <= delta, d, delta * np.sign(d))
            else:
                r = y - F
            idx = rng.choice(n, int(self.subsample * n), replace=False) if self.subsample < 1 else np.arange(n)
            t = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf, random_state=rng.randint(1 << 30))
            t.fit(X[idx], r[idx])
            F += self.learning_rate * t.predict(X)
            self.estimators_.append(t)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        X = check_array(X)
        F = np.full(len(X), self._F0)
        for t in self.estimators_:
            F += self.learning_rate * t.predict(X)
        return F


class HistGradientBoostingClassifier(GradientBoostingClassifier):
    def __init__(self, learning_rate=0.1, max_iter=100, max_depth=None, max_leaf_nodes=31,
                 min_samples_leaf=20, l2_regularization=0.0, random_state=None):
        super().__init__(learning_rate=learning_rate, n_estimators=max_iter, max_depth=max_depth or 3,
                         min_samples_leaf=min_samples_leaf, random_state=random_state)
        self.max_iter = max_iter; self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization


class HistGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self, learning_rate=0.1, max_iter=100, max_depth=None, max_leaf_nodes=31,
                 min_samples_leaf=20, l2_regularization=0.0, random_state=None):
        super().__init__(learning_rate=learning_rate, n_estimators=max_iter, max_depth=max_depth or 3,
                         min_samples_leaf=min_samples_leaf, random_state=random_state)
        self.max_iter = max_iter


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
        self.estimator = estimator; self.n_estimators = n_estimators
        self.learning_rate = learning_rate; self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoostClassifier supports binary only in salearn 0.1")
        n = len(X)
        w = np.ones(n) / n if sample_weight is None else np.asarray(sample_weight, dtype=np.float64) / np.sum(sample_weight)
        yb = np.where(y == self.classes_[1], 1, -1)
        self.estimators_ = []; self.estimator_weights_ = []
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_estimators):
            base = clone(self.estimator) if self.estimator is not None else DecisionTreeClassifier(max_depth=1)
            if hasattr(base, "random_state"):
                try:
                    base.set_params(random_state=int(rng.randint(1 << 30)))
                except Exception:
                    pass
            try:
                base.fit(X, y, sample_weight=w)
            except TypeError:
                base.fit(X, y)
            pred = base.predict(X)
            pb = np.where(pred == self.classes_[1], 1, -1)
            err = (w[pb != yb]).sum() / w.sum()
            err = min(max(err, 1e-12), 1 - 1e-12)
            a = self.learning_rate * 0.5 * np.log((1 - err) / err)
            w *= np.exp(-a * yb * pb)
            w /= w.sum()
            self.estimators_.append(base)
            self.estimator_weights_.append(a)
        self.estimator_weights_ = np.array(self.estimator_weights_)
        self._is_fitted = True
        return self

    def decision_function(self, X):
        X = check_array(X)
        s = np.zeros(len(X))
        for a, e in zip(self.estimator_weights_, self.estimators_):
            pb = np.where(e.predict(X) == self.classes_[1], 1, -1)
            s += a * pb
        return s

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        s = self.decision_function(X)
        p1 = 1 / (1 + np.exp(-2 * s))
        return np.c_[1 - p1, p1]


class AdaBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0, loss="linear", random_state=None):
        self.estimator = estimator; self.n_estimators = n_estimators
        self.learning_rate = learning_rate; self.loss = loss; self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        y = np.asarray(y, dtype=np.float64)
        n = len(X)
        w = np.ones(n) / n if sample_weight is None else np.asarray(sample_weight) / np.sum(sample_weight)
        self.estimators_ = []; self.estimator_weights_ = []
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_estimators):
            base = clone(self.estimator) if self.estimator is not None else DecisionTreeRegressor(max_depth=3)
            try:
                base.fit(X, y, sample_weight=w)
            except TypeError:
                base.fit(X, y)
            pred = base.predict(X)
            err = np.abs(pred - y)
            mx = err.max()
            err = err / mx if mx > 0 else err
            e = (w * err).sum() / w.sum()
            e = min(max(e, 1e-12), 1 - 1e-12)
            beta = e / (1 - e)
            a = self.learning_rate * np.log(1 / beta)
            w *= np.power(beta, self.learning_rate * (1 - err))
            w /= w.sum()
            self.estimators_.append(base)
            self.estimator_weights_.append(a)
        self.estimator_weights_ = np.array(self.estimator_weights_)
        self._is_fitted = True
        return self

    def predict(self, X):
        X = check_array(X)
        P = np.array([e.predict(X) for e in self.estimators_])
        # weighted median
        out = np.empty(len(X))
        for i in range(len(X)):
            order = np.argsort(P[:, i])
            cumsum = np.cumsum(self.estimator_weights_[order])
            out[i] = P[order[np.searchsorted(cumsum, cumsum[-1] / 2)], i]
        return out


class BaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0,
                 bootstrap=True, n_jobs=None, random_state=None):
        self.estimator = estimator; self.n_estimators = n_estimators
        self.max_samples = max_samples; self.max_features = max_features
        self.bootstrap = bootstrap; self.n_jobs = n_jobs; self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n, p = X.shape
        ns = int(self.max_samples * n) if isinstance(self.max_samples, float) else self.max_samples
        nf = int(self.max_features * p) if isinstance(self.max_features, float) else self.max_features
        rng = np.random.RandomState(self.random_state)
        jobs = []
        self.features_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, ns, replace=self.bootstrap)
            fidx = rng.choice(p, nf, replace=False)
            self.features_.append(fidx)
            base = clone(self.estimator) if self.estimator is not None else DecisionTreeClassifier()
            jobs.append((base, X[np.ix_(idx, fidx)], y[idx], None, rng.randint(1 << 30)))
        with ThreadPoolExecutor(max_workers=_resolve_n_jobs(self.n_jobs)) as ex:
            self.estimators_ = list(ex.map(lambda a: _fit_one(*a), jobs))
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        X = check_array(X)
        P = np.mean([e.predict_proba(X[:, f]) if hasattr(e, "predict_proba") else np.eye(len(self.classes_))[np.searchsorted(self.classes_, e.predict(X[:, f]))] for e, f in zip(self.estimators_, self.features_)], axis=0)
        return P

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class BaggingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0,
                 bootstrap=True, n_jobs=None, random_state=None):
        self.estimator = estimator; self.n_estimators = n_estimators
        self.max_samples = max_samples; self.max_features = max_features
        self.bootstrap = bootstrap; self.n_jobs = n_jobs; self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n, p = X.shape
        ns = int(self.max_samples * n) if isinstance(self.max_samples, float) else self.max_samples
        nf = int(self.max_features * p) if isinstance(self.max_features, float) else self.max_features
        rng = np.random.RandomState(self.random_state)
        self.features_ = [rng.choice(p, nf, replace=False) for _ in range(self.n_estimators)]
        jobs = []
        for i in range(self.n_estimators):
            idx = rng.choice(n, ns, replace=self.bootstrap)
            base = clone(self.estimator) if self.estimator is not None else DecisionTreeRegressor()
            jobs.append((base, X[np.ix_(idx, self.features_[i])], np.asarray(y)[idx], None, rng.randint(1 << 30)))
        with ThreadPoolExecutor(max_workers=_resolve_n_jobs(self.n_jobs)) as ex:
            self.estimators_ = list(ex.map(lambda a: _fit_one(*a), jobs))
        self._is_fitted = True
        return self

    def predict(self, X):
        X = check_array(X)
        return np.mean([e.predict(X[:, f]) for e, f in zip(self.estimators_, self.features_)], axis=0)


class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting="hard", weights=None):
        self.estimators = estimators; self.voting = voting; self.weights = weights

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for n, e in self.estimators:
            c = clone(e)
            try:
                c.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                c.fit(X, y)
            self.estimators_.append((n, c))
        self._is_fitted = True
        return self

    def predict(self, X):
        X = check_array(X)
        if self.voting == "hard":
            P = np.array([e.predict(X) for _, e in self.estimators_])
            out = []
            for j in range(len(X)):
                v, c = np.unique(P[:, j], return_counts=True)
                if self.weights is not None:
                    # weighted vote
                    wv = {}
                    for k, e_pred in enumerate(P[:, j]):
                        wv[e_pred] = wv.get(e_pred, 0) + self.weights[k]
                    out.append(max(wv, key=wv.get))
                else:
                    out.append(v[c.argmax()])
            return np.array(out)
        # soft
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_proba(self, X):
        X = check_array(X)
        P = np.array([e.predict_proba(X) for _, e in self.estimators_])
        # align classes
        aligned = []
        for (n, e), p in zip(self.estimators_, P):
            if np.array_equal(e.classes_, self.classes_):
                aligned.append(p)
            else:
                a = np.zeros((len(X), len(self.classes_)))
                for j, c in enumerate(e.classes_):
                    a[:, list(self.classes_).index(c)] = p[:, j]
                aligned.append(a)
        A = np.array(aligned)
        if self.weights is not None:
            w = np.asarray(self.weights)[:, None, None]
            return (A * w).sum(0) / np.sum(self.weights)
        return A.mean(0)


class VotingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, weights=None):
        self.estimators = estimators; self.weights = weights

    def fit(self, X, y, sample_weight=None):
        self.estimators_ = []
        for n, e in self.estimators:
            c = clone(e)
            try:
                c.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                c.fit(X, y)
            self.estimators_.append((n, c))
        self._is_fitted = True
        return self

    def predict(self, X):
        X = check_array(X)
        P = np.array([e.predict(X) for _, e in self.estimators_])
        if self.weights is not None:
            return np.average(P, axis=0, weights=self.weights)
        return P.mean(0)


class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, final_estimator=None, cv=5):
        self.estimators = estimators; self.final_estimator = final_estimator; self.cv = cv

    def fit(self, X, y):
        from .model_selection import check_cv
        from .utils import safe_indexing
        X = np.asarray(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        cv = check_cv(self.cv, y, classifier=True)
        n = len(X)
        # out-of-fold predictions for meta features
        meta = np.zeros((n, len(self.estimators) * (len(self.classes_) if len(self.classes_) > 2 else 1)))
        for tr, te in cv.split(X, y):
            for j, (nm, e) in enumerate(self.estimators):
                c = clone(e).fit(safe_indexing(X, tr), safe_indexing(y, tr))
                if hasattr(c, "predict_proba"):
                    p = c.predict_proba(safe_indexing(X, te))
                    if p.shape[1] == 2 and len(self.classes_) == 2:
                        p = p[:, 1:2]
                    meta[te, j * p.shape[1]:(j + 1) * p.shape[1]] = p
                else:
                    pr = c.predict(safe_indexing(X, te))
                    meta[te, j] = np.searchsorted(self.classes_, pr)
        from .linear_model import LogisticRegression
        self.final_estimator_ = clone(self.final_estimator) if self.final_estimator is not None else LogisticRegression()
        self.final_estimator_.fit(meta, y)
        # refit bases
        self.estimators_ = [(nm, clone(e).fit(X, y)) for nm, e in self.estimators]
        self._is_fitted = True
        return self

    def _meta(self, X):
        X = np.asarray(X)
        parts = []
        for _, e in self.estimators_:
            if hasattr(e, "predict_proba"):
                p = e.predict_proba(X)
                if p.shape[1] == 2 and len(self.classes_) == 2:
                    p = p[:, 1:2]
                parts.append(p)
            else:
                parts.append(np.searchsorted(self.classes_, e.predict(X)).reshape(-1, 1))
        return np.hstack(parts)

    def predict(self, X):
        return self.final_estimator_.predict(self._meta(np.asarray(X)))

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(self._meta(np.asarray(X)))


class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, final_estimator=None, cv=5):
        self.estimators = estimators; self.final_estimator = final_estimator; self.cv = cv

    def fit(self, X, y):
        from .model_selection import check_cv
        from .utils import safe_indexing
        X = np.asarray(X); y = np.asarray(y)
        cv = check_cv(self.cv)
        meta = np.zeros((len(X), len(self.estimators)))
        for tr, te in cv.split(X, y):
            for j, (nm, e) in enumerate(self.estimators):
                c = clone(e).fit(safe_indexing(X, tr), safe_indexing(y, tr))
                meta[te, j] = c.predict(safe_indexing(X, te))
        from .linear_model import Ridge
        self.final_estimator_ = clone(self.final_estimator) if self.final_estimator is not None else Ridge()
        self.final_estimator_.fit(meta, y)
        self.estimators_ = [(nm, clone(e).fit(X, y)) for nm, e in self.estimators]
        self._is_fitted = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        meta = np.column_stack([e.predict(X) for _, e in self.estimators_])
        return self.final_estimator_.predict(meta)
