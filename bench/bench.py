"""Benchmarks: salearn vs sklearn."""
from __future__ import annotations

import time
import numpy as np


def bench(fn, *a, n=3, **k):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*a, **k)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    from sklearn.datasets import make_classification, make_regression
    Xc, yc = make_classification(n_samples=5000, n_features=50, random_state=0)
    Xr, yr = make_regression(n_samples=5000, n_features=50, random_state=0)

    import salearn.linear_model as L
    import sklearn.linear_model as SL

    t1 = bench(SL.Ridge().fit, Xc, yc)
    t2 = bench(L.Ridge().fit, Xc, yc)
    print(f"Ridge        sklearn={t1:.3f}s  salearn={t2:.3f}s  speedup={t1 / max(t2, 1e-9):.2f}x")

    import salearn.ensemble as E
    import sklearn.ensemble as SE
    t1 = bench(SE.RandomForestClassifier(n_estimators=20, n_jobs=-1).fit, Xc, yc, n=1)
    t2 = bench(E.RandomForestClassifier(n_estimators=20, n_jobs=-1).fit, Xc, yc, n=1)
    print(f"RandomForest sklearn={t1:.3f}s  salearn={t2:.3f}s  speedup={t1 / max(t2, 1e-9):.2f}x")

    import salearn.cluster as C
    import sklearn.cluster as SC
    t1 = bench(SC.KMeans(n_clusters=10, n_init=3).fit, Xc, n=1)
    t2 = bench(C.KMeans(n_clusters=10, n_init=3).fit, Xc, n=1)
    print(f"KMeans       sklearn={t1:.3f}s  salearn={t2:.3f}s  speedup={t1 / max(t2, 1e-9):.2f}x")

    import salearn.neighbors as N
    import sklearn.neighbors as SN
    t1 = bench(SN.KNeighborsClassifier(n_neighbors=5).fit(Xc, yc).predict, Xc[:500])
    t2 = bench(N.KNeighborsClassifier(n_neighbors=5).fit(Xc, yc).predict, Xc[:500])
    print(f"KNN predict  sklearn={t1:.3f}s  salearn={t2:.3f}s  speedup={t1 / max(t2, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
