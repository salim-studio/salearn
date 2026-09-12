"""Smoke tests for salearn — sklearn compat."""
from __future__ import annotations

import numpy as np


def _data_cls(n=200, p=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.2 * rng.randn(n) > 0).astype(int)
    return X, y


def _data_reg(n=200, p=10, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = X @ np.arange(p) + rng.randn(n)
    return X, y


def test_linear():
    from salearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    X, y = _data_reg()
    for M in (LinearRegression(), Ridge(), Lasso(alpha=0.1)):
        M.fit(X, y)
        assert M.predict(X).shape == (len(X),)
        assert np.isfinite(M.score(X, y))
    Xc, yc = _data_cls()
    m = LogisticRegression(max_iter=50).fit(Xc, yc)
    assert set(np.unique(m.predict(Xc))) <= {0, 1}
    assert m.predict_proba(Xc).shape == (len(Xc), 2)


def test_tree_forest():
    from salearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from salearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    Xc, yc = _data_cls()
    DecisionTreeClassifier().fit(Xc, yc).predict(Xc)
    RandomForestClassifier(n_estimators=5, random_state=0).fit(Xc, yc).predict(Xc)
    GradientBoostingClassifier(n_estimators=5).fit(Xc, yc).predict(Xc)
    Xr, yr = _data_reg()
    DecisionTreeRegressor().fit(Xr, yr).predict(Xr)
    RandomForestRegressor(n_estimators=5, random_state=0).fit(Xr, yr).predict(Xr)


def test_neighbors_nb():
    from salearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from salearn.naive_bayes import GaussianNB, MultinomialNB
    Xc, yc = _data_cls()
    KNeighborsClassifier(n_neighbors=3).fit(Xc, yc).predict(Xc)
    KNeighborsRegressor().fit(Xc, yc).predict(Xc)
    GaussianNB().fit(Xc, yc).predict(Xc)
    Xm = np.abs(Xc) + 0.1
    MultinomialNB().fit(Xm, yc).predict(Xm)


def test_cluster_decomp():
    from salearn.cluster import KMeans, DBSCAN
    from salearn.decomposition import PCA, TruncatedSVD
    Xc, _ = _data_cls(n=150)
    k = KMeans(n_clusters=2, n_init=2, random_state=0).fit(Xc)
    assert len(np.unique(k.labels_)) == 2
    DBSCAN(eps=2.0, min_samples=3).fit(Xc)
    assert PCA(n_components=2).fit_transform(Xc).shape == (150, 2)
    assert TruncatedSVD(n_components=2).fit_transform(Xc).shape == (150, 2)


def test_preprocessing_pipeline_cv():
    from salearn.preprocessing import StandardScaler, OneHotEncoder
    from salearn.pipeline import make_pipeline
    from salearn.linear_model import LogisticRegression
    from salearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    Xc, yc = _data_cls()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=50))
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
    pipe.fit(Xtr, ytr)
    assert 0 <= pipe.score(Xte, yte) <= 1
    assert len(cross_val_score(pipe, Xc, yc, cv=3)) == 3
    g = GridSearchCV(LogisticRegression(max_iter=30), {"C": [0.1, 1.0]}, cv=2).fit(Xc, yc)
    assert "C" in g.best_params_


def test_metrics_svm_mixture_mlp():
    from salearn import metrics as M
    from salearn.svm import LinearSVC
    from salearn.mixture import GaussianMixture
    from salearn.neural_network import MLPClassifier
    Xc, yc = _data_cls()
    assert 0 <= M.accuracy_score(yc, yc) <= 1
    assert M.r2_score(yc, yc) == 1.0
    LinearSVC(max_iter=200).fit(Xc, yc).predict(Xc)
    GaussianMixture(n_components=2, max_iter=5).fit(Xc).predict(Xc)
    MLPClassifier(hidden_layer_sizes=(10,), max_iter=5).fit(Xc, yc).predict(Xc)
