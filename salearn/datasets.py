"""salearn.datasets — toy datasets (sklearn-compatible loaders)."""
from __future__ import annotations

import numpy as np
from collections import namedtuple

Bunch = namedtuple("Bunch", ["data", "target", "feature_names", "target_names", "DESCR"])


def make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,
                        n_classes=2, n_clusters_per_class=2, random_state=None, **kw):
    from sklearn.datasets import make_classification as _m
    X, y = _m(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
              n_redundant=n_redundant, n_classes=n_classes,
              n_clusters_per_class=n_clusters_per_class, random_state=random_state)
    return X, y


def make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.1, random_state=None, **kw):
    from sklearn.datasets import make_regression as _m
    return _m(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
              noise=noise, random_state=random_state)


def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=None, **kw):
    from sklearn.datasets import make_blobs as _m
    return _m(n_samples=n_samples, n_features=n_features, centers=centers,
              cluster_std=cluster_std, random_state=random_state)


def make_moons(n_samples=100, noise=0.1, random_state=None):
    from sklearn.datasets import make_moons as _m
    return _m(n_samples=n_samples, noise=noise, random_state=random_state)


def make_circles(n_samples=100, noise=0.1, factor=0.8, random_state=None):
    from sklearn.datasets import make_circles as _m
    return _m(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)


def load_iris(return_X_y=False):
    from sklearn.datasets import load_iris as _l
    d = _l()
    if return_X_y:
        return d.data, d.target
    return d


def load_digits(return_X_y=False):
    from sklearn.datasets import load_digits as _l
    d = _l()
    if return_X_y:
        return d.data, d.target
    return d


def load_wine(return_X_y=False):
    from sklearn.datasets import load_wine as _l
    d = _l()
    if return_X_y:
        return d.data, d.target
    return d


def load_breast_cancer(return_X_y=False):
    from sklearn.datasets import load_breast_cancer as _l
    d = _l()
    if return_X_y:
        return d.data, d.target
    return d


def load_diabetes(return_X_y=False):
    from sklearn.datasets import load_diabetes as _l
    d = _l()
    if return_X_y:
        return d.data, d.target
    return d


def fetch_california_housing(return_X_y=False):
    from sklearn.datasets import fetch_california_housing as _f
    d = _f()
    if return_X_y:
        return d.data, d.target
    return d
