"""salearn.model_selection — splits, CV, search (sklearn-compatible)."""
from __future__ import annotations

import itertools
import numpy as np
from .base import clone
from .utils import safe_indexing


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None,
                     shuffle=True, stratify=None):
    n = len(arrays[0])
    for a in arrays:
        if len(a) != n:
            raise ValueError("inconsistent lengths")
    rng = np.random.RandomState(random_state)
    idx = np.arange(n)
    if shuffle:
        if stratify is not None:
            stratify = np.asarray(stratify)
            # stratified shuffle: permute within each class then interleave proportionally
            idx = _stratified_shuffle(stratify, test_size, train_size, rng)
            train_idx, test_idx = idx
            out = []
            for a in arrays:
                A = np.asarray(a) if not hasattr(a, "iloc") else a
                out += [safe_indexing(A, train_idx), safe_indexing(A, test_idx)]
            return out
        else:
            rng.shuffle(idx)
    if test_size is None and train_size is None:
        test_size = 0.25
    if test_size is not None:
        n_test = int(test_size * n) if isinstance(test_size, float) else int(test_size)
    else:
        n_train = int(train_size * n) if isinstance(train_size, float) else int(train_size)
        n_test = n - n_train
    test_idx = idx[-n_test:] if n_test else np.array([], dtype=int)
    train_idx = idx[:n - n_test]
    out = []
    for a in arrays:
        A = a
        out += [safe_indexing(A, train_idx), safe_indexing(A, test_idx)]
    return out


def _stratified_shuffle(y, test_size, train_size, rng):
    classes, inv = np.unique(y, return_inverse=True)
    per_class = [np.where(inv == c)[0] for c in range(len(classes))]
    for p in per_class:
        rng.shuffle(p)
    if test_size is None and train_size is None:
        test_size = 0.25
    train_idx, test_idx = [], []
    for p in per_class:
        n_test = int(len(p) * test_size) if isinstance(test_size, float) else (len(p) - int(train_size * len(p) / len(y)) if train_size is not None else int(test_size / len(y) * len(p)))
        n_test = max(min(n_test, len(p) - 1), 0) if len(p) > 1 else 0
        test_idx += p[:n_test].tolist()
        train_idx += p[n_test:].tolist()
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits; self.shuffle = shuffle; self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(idx)
        sizes = np.full(self.n_splits, n // self.n_splits)
        sizes[:n % self.n_splits] += 1
        s = 0
        for sz in sizes:
            test = idx[s:s + sz]
            train = np.concatenate([idx[:s], idx[s + sz:]])
            yield train, test
            s += sz

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits; self.shuffle = shuffle; self.random_state = random_state

    def split(self, X, y, groups=None):
        y = np.asarray(y)
        classes, inv = np.unique(y, return_inverse=True)
        per_class = [np.where(inv == c)[0] for c in range(len(classes))]
        rng = np.random.RandomState(self.random_state)
        if self.shuffle:
            for p in per_class:
                rng.shuffle(p)
        folds = [[] for _ in range(self.n_splits)]
        for p in per_class:
            for i, ix in enumerate(p):
                folds[i % self.n_splits].append(ix)
        for k in range(self.n_splits):
            test = np.array(sorted(folds[k]))
            train = np.array(sorted([x for j, f in enumerate(folds) if j != k for x in f]))
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class ShuffleSplit:
    def __init__(self, n_splits=5, test_size=0.2, random_state=None):
        self.n_splits = n_splits; self.test_size = test_size; self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        n_test = int(self.test_size * n) if isinstance(self.test_size, float) else self.test_size
        for _ in range(self.n_splits):
            idx = rng.permutation(n)
            yield idx[n_test:], idx[:n_test]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class LeaveOneOut:
    def split(self, X, y=None, groups=None):
        n = len(X)
        for i in range(n):
            yield np.delete(np.arange(n), i), np.array([i])

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(X)


def check_cv(cv=5, y=None, classifier=False):
    if hasattr(cv, "split"):
        return cv
    if classifier:
        return StratifiedKFold(n_splits=cv)
    return KFold(n_splits=cv)


def cross_val_score(estimator, X, y=None, cv=5, scoring=None, groups=None, n_jobs=None):
    from .metrics import get_scorer
    X = np.asarray(X) if not hasattr(X, "iloc") else X
    y = np.asarray(y) if y is not None else None
    clf = getattr(estimator, "_estimator_type", None) == "classifier"
    cv = check_cv(cv, y, classifier=clf)
    scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
    scores = []
    for tr, te in cv.split(X, y, groups):
        est = clone(estimator)
        Xt, Xte = safe_indexing(X, tr), safe_indexing(X, te)
        yt = None if y is None else safe_indexing(y, tr)
        yte = None if y is None else safe_indexing(y, te)
        est.fit(Xt, yt)
        if scorer is None:
            scores.append(est.score(Xte, yte))
        else:
            scores.append(scorer(est, Xte, yte))
    return np.array(scores)


def cross_validate(estimator, X, y=None, cv=5, scoring=None, return_train_score=False):
    test = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return {"test_score": test, "fit_time": np.zeros_like(test), "score_time": np.zeros_like(test)}


def cross_val_predict(estimator, X, y, cv=5):
    X = np.asarray(X); y = np.asarray(y)
    clf = getattr(estimator, "_estimator_type", None) == "classifier"
    cv = check_cv(cv, y, classifier=clf)
    out = np.empty(len(X), dtype=np.asarray(y).dtype if y is not None else np.float64)
    for tr, te in cv.split(X, y):
        est = clone(estimator).fit(safe_indexing(X, tr), safe_indexing(y, tr))
        out[te] = est.predict(safe_indexing(X, te))
    return out


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring=None, refit=True, n_jobs=None, verbose=0):
        self.estimator = estimator; self.param_grid = param_grid
        self.cv = cv; self.scoring = scoring; self.refit = refit
        self.n_jobs = n_jobs; self.verbose = verbose

    def _grid(self):
        if isinstance(self.param_grid, dict):
            keys = list(self.param_grid)
            for vals in itertools.product(*[self.param_grid[k] for k in keys]):
                yield dict(zip(keys, vals))
        else:
            for d in self.param_grid:
                keys = list(d)
                for vals in itertools.product(*[d[k] for k in keys]):
                    yield dict(zip(keys, vals))

    def fit(self, X, y=None, groups=None):
        from .metrics import get_scorer
        X = np.asarray(X) if not hasattr(X, "iloc") else X
        scorer = get_scorer(self.scoring) if isinstance(self.scoring, str) else self.scoring
        clf = getattr(self.estimator, "_estimator_type", None) == "classifier"
        cv = check_cv(self.cv, y, classifier=clf)
        results = []
        best = None
        for params in self._grid():
            est = clone(self.estimator).set_params(**params)
            scores = []
            for tr, te in cv.split(X, y, groups):
                e = clone(est)
                e.fit(safe_indexing(X, tr), None if y is None else safe_indexing(y, tr))
                Xt, yt = safe_indexing(X, te), None if y is None else safe_indexing(y, te)
                s = e.score(Xt, yt) if scorer is None else scorer(e, Xt, yt)
                scores.append(s)
            mean = float(np.mean(scores))
            results.append((mean, params, scores))
            if best is None or mean > best[0]:
                best = (mean, params, scores)
        self.cv_results_ = results
        self.best_score_ = best[0]; self.best_params_ = best[1]
        self.best_index_ = int(np.argmax([r[0] for r in results]))
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
            for a in ("classes_", "coef_", "intercept_", "feature_importances_"):
                if hasattr(self.best_estimator_, a):
                    setattr(self, a, getattr(self.best_estimator_, a))
            self._is_fitted = True
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


class RandomizedSearchCV(GridSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, scoring=None,
                 refit=True, random_state=None, n_jobs=None, verbose=0):
        super().__init__(estimator, {}, cv=cv, scoring=scoring, refit=refit, n_jobs=n_jobs, verbose=verbose)
        self.param_distributions = param_distributions; self.n_iter = n_iter
        self.random_state = random_state

    def _grid(self):
        rng = np.random.RandomState(self.random_state)
        keys = list(self.param_distributions)
        for _ in range(self.n_iter):
            d = {}
            for k in keys:
                v = self.param_distributions[k]
                if hasattr(v, "rvs"):
                    d[k] = v.rvs(random_state=rng)
                elif isinstance(v, (list, tuple, np.ndarray)):
                    d[k] = v[rng.randint(len(v))]
                else:
                    d[k] = v
            yield d


def learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, scoring=None):
    X = np.asarray(X); y = np.asarray(y)
    clf = getattr(estimator, "_estimator_type", None) == "classifier"
    cv = check_cv(cv, y, classifier=clf)
    ts = (train_sizes * len(X)).astype(int) if train_sizes.max() <= 1 else train_sizes.astype(int)
    tr_scores, te_scores = [], []
    for n in ts:
        tr_s, te_s = [], []
        for tr, te in cv.split(X, y):
            tr = tr[:max(n, 1)]
            e = clone(estimator).fit(X[tr], y[tr])
            if scoring is None:
                tr_s.append(e.score(X[tr], y[tr])); te_s.append(e.score(X[te], y[te]))
            else:
                from .metrics import get_scorer
                sc = get_scorer(scoring) if isinstance(scoring, str) else scoring
                tr_s.append(sc(e, X[tr], y[tr])); te_s.append(sc(e, X[te], y[te]))
        tr_scores.append(tr_s); te_scores.append(te_s)
    return np.array(ts), np.array(tr_scores), np.array(te_scores)


def validation_curve(estimator, X, y, param_name, param_range, cv=5, scoring=None):
    X = np.asarray(X); y = np.asarray(y)
    tr_scores, te_scores = [], []
    for v in param_range:
        e = clone(estimator).set_params(**{param_name: v})
        te = cross_val_score(e, X, y, cv=cv, scoring=scoring)
        e.fit(X, y)
        tr = e.score(X, y) if scoring is None else None
        tr_scores.append([tr if tr is not None else te.mean()])
        te_scores.append(te)
    return np.asarray(param_range), np.array(tr_scores), np.array(te_scores)
