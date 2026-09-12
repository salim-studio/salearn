"""salearn.automl — 5-line AutoML for developers & analysts.

    from salearn.automl import find_best, leaderboard
    best, report = find_best(X, y, task="auto", time_budget=10)  # seconds-ish
    print(report)

- tries a smart portfolio (logistic/ridge/RF/ET/GBM/KNN/NB/MLP...) with CV
- task auto-detect (classification vs regression)
- deterministic, no new deps, respects time_budget loosely
"""
from __future__ import annotations

import time
import numpy as np
from .base import clone


def _detect_task(y) -> str:
    y = np.asarray(y)
    if y.dtype.kind in "OUS" or len(np.unique(y)) <= 20 and y.dtype.kind in "iu":
        # heuristic: few unique ints/strings -> classification
        if y.dtype.kind in "OUS" or len(np.unique(y)) <= max(10, len(y) * 0.05):
            return "classification"
    return "regression"


def _portfolio(task: str, n: int, p: int):
    if task == "classification":
        from .linear_model import LogisticRegression
        from .tree import DecisionTreeClassifier
        from .ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
        from .neighbors import KNeighborsClassifier
        from .naive_bayes import GaussianNB
        from .neural_network import MLPClassifier
        ests = [
            ("logreg", LogisticRegression(max_iter=200)),
            ("tree", DecisionTreeClassifier()),
            ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=100, n_jobs=-1)),
            ("knn", KNeighborsClassifier(n_neighbors=min(7, max(1, n - 1)))),
            ("gnb", GaussianNB()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=200)),
        ]
        if n < 5000:
            ests.append(("gbm", GradientBoostingClassifier(n_estimators=100)))
            ests.append(("hgb", HistGradientBoostingClassifier(max_iter=100)))
        return ests
    from .linear_model import Ridge, Lasso, ElasticNet, LinearRegression
    from .tree import DecisionTreeRegressor
    from .ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    from .neighbors import KNeighborsRegressor
    from .neural_network import MLPRegressor
    ests = [
        ("ridge", Ridge()),
        ("lasso", Lasso(alpha=0.01)),
        ("tree", DecisionTreeRegressor()),
        ("rf", RandomForestRegressor(n_estimators=100, n_jobs=-1)),
        ("et", ExtraTreesRegressor(n_estimators=100, n_jobs=-1)),
        ("knn", KNeighborsRegressor(n_neighbors=min(7, max(1, n - 1)))),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64,), max_iter=200)),
    ]
    if n < 5000:
        ests.append(("gbm", GradientBoostingRegressor(n_estimators=100)))
    return ests


def leaderboard(X, y, task="auto", cv=3, scoring=None, time_budget=30, verbose=False):
    """Evaluate portfolio with CV. Returns ranked list of dicts."""
    from .model_selection import cross_val_score
    X = np.asarray(X)
    y = np.asarray(y)
    task = _detect_task(y) if task == "auto" else task
    rows = []
    t0 = time.time()
    for name, est in _portfolio(task, len(X), X.shape[1] if X.ndim > 1 else 1):
        if time.time() - t0 > time_budget:
            break
        try:
            t1 = time.time()
            s = cross_val_score(clone(est), X, y, cv=cv, scoring=scoring)
            rows.append({"model": name, "estimator": clone(est),
                         "mean": float(np.mean(s)), "std": float(np.std(s)),
                         "fit_time": round(time.time() - t1, 3)})
            if verbose:
                print(f"{name}: {np.mean(s):.4f}±{np.std(s):.4f}")
        except Exception as e:
            if verbose:
                print(f"{name} failed: {e}")
    rows.sort(key=lambda r: -r["mean"])
    return rows


def find_best(X, y, task="auto", cv=3, scoring=None, time_budget=30, refit=True, verbose=False):
    """Return (best_estimator_fitted, leaderboard_rows)."""
    rows = leaderboard(X, y, task=task, cv=cv, scoring=scoring, time_budget=time_budget, verbose=verbose)
    if not rows:
        raise RuntimeError("All AutoML candidates failed")
    best = rows[0]["estimator"]
    if refit:
        best.fit(np.asarray(X), np.asarray(y))
    return best, rows


__all__ = ["find_best", "leaderboard"]
