"""Extended tests for salearn 0.2.0 — DB, IO, EDA, time-series, NLP, AutoML, persist."""
from __future__ import annotations

import numpy as np


def test_db_sqlite_memory():
    from salearn.db import Database, QueryBuilder
    db = Database.sqlite_memory()
    db.execute("CREATE TABLE t (a REAL, b REAL, label INT)")
    db.executemany("INSERT INTO t VALUES (?,?,?)", [(1.0, 2.0, 0), (5.0, 6.0, 1), (1.5, 2.5, 0)])
    assert db.count("t") == 3
    assert "t" in db.list_tables()
    q = QueryBuilder("t").select("a", "label").where("a > 1").limit(10).build()
    rows, cols = db.query(q)
    assert len(rows) == 2
    X, y, feats = db.load_X_y("SELECT * FROM t", target="label")
    assert X.shape == (3, 2) and len(y) == 3
    Xtr, Xte, ytr, yte, _ = db.train_test_from_sql("SELECT * FROM t", target="label", test_size=1)
    assert len(Xte) == 1
    db.close()


def test_io_eda_clean():
    from salearn import io as SIO
    from salearn import eda, clean
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("a,b,label\n1,2,0\n3,,1\n5,6,1\n")
        data = SIO.load_csv(path)
        try:
            import pandas as pd
            assert len(data) == 3
            rep = eda.profile(data)
            assert rep["n_rows"] == 3
            Xc = clean.DataCleaner().fit_transform(data)
            assert len(Xc) >= 2
        except ImportError:
            rows, cols = data
            assert len(rows) == 3
    finally:
        os.remove(path)


def test_timeseries():
    from salearn.timeseries import NaiveForecaster, ARForecaster, temporal_split, rolling_origin_cv
    y = np.arange(30, dtype=float) + np.sin(np.arange(30))
    tr, te = temporal_split(y, test_size=7)
    assert len(te) == 7
    p = NaiveForecaster().fit(tr).predict(7)
    assert len(p) == 7 and p[0] == tr[-1]
    p2 = ARForecaster(lags=3).fit(tr).predict(5)
    assert len(p2) == 5 and np.all(np.isfinite(p2))
    scores = rolling_origin_cv(y, NaiveForecaster(), horizon=3, folds=2)
    assert len(scores) >= 1


def test_text_arabic():
    from salearn.text import clean_text, TfidfVectorizer, TextClassifier
    assert "ا" in clean_text("أإآ السلام عليكم")
    docs = ["هذا فيلم رائع جدا", "فيلم سيء وممل", "رائع وممتع", "سيء جدا وممل"]
    y = [1, 0, 1, 0]
    clf = TextClassifier(max_features=50, max_iter=100).fit(docs, y)
    pred = clf.predict(["فيلم رائع", "ممل وسيء"])
    assert set(pred) <= {0, 1}
    X = TfidfVectorizer(max_features=20).fit_transform(docs)
    assert X.shape[0] == 4


def test_automl_explain_persist(tmp_path=None):
    from salearn.automl import find_best
    from salearn.explain import model_report
    from salearn.persist import save_model, load_model
    import tempfile, os
    rng = np.random.RandomState(0)
    X = rng.randn(60, 5)
    y = (X[:, 0] > 0).astype(int)
    best, rows = find_best(X, y, cv=2, time_budget=20, verbose=False)
    assert rows and hasattr(best, "predict")
    rep = model_report(best, X, y)
    assert "accuracy" in rep
    fd, path = tempfile.mkstemp(suffix=".ssl")
    os.close(fd)
    try:
        save_model(best, path)
        m2 = load_model(path)
        assert len(m2.predict(X)) == len(X)
    finally:
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(path + ".json"):
            os.remove(path + ".json")


def test_features_viz():
    from salearn.features import TargetEncoder, NumericBinner, auto_featurize
    import numpy as np
    try:
        import pandas as pd
        df = pd.DataFrame({"c": list("aababbaa"), "x": np.arange(8, dtype=float),
                           "y": [0, 1, 0, 1, 0, 1, 0, 1]})
        te = TargetEncoder(columns=["c"]).fit(df[["c"]], df["y"])
        assert len(te.transform(df[["c"]])) == 8
        X, y, names = auto_featurize(df, target="y")
        assert X.shape[0] == 8
    except ImportError:
        pass
    b = NumericBinner(n_bins=3).fit(np.arange(10).reshape(-1, 1))
    assert b.transform(np.arange(10).reshape(-1, 1)).shape == (10, 1)
