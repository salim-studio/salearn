"""salearn end-to-end example: DB -> EDA -> Clean -> AutoML -> Explain -> Persist + TS + Arabic NLP."""
import numpy as np

print("== 1) DB ==")
from salearn.db import Database
db = Database.sqlite_memory()
rng = np.random.RandomState(0)
n = 200
df_rows = [(float(a), float(b), int((a + b + rng.randn() * 0.5) > 0))
           for a, b in rng.randn(n, 2)]
db.execute("CREATE TABLE demo (a REAL, b REAL, label INT)")
db.executemany("INSERT INTO demo VALUES (?,?,?)", df_rows)
print("tables:", db.list_tables(), "count:", db.count("demo"))
X, y, feats = db.load_X_y("SELECT * FROM demo", target="label")
print("X", X.shape, "feats", feats)

print("== 2) EDA + Clean ==")
from salearn import eda
from salearn.clean import DataCleaner
print({k: v for k, v in list(eda.profile((db.query("SELECT * FROM demo")[0], ["a", "b", "label"]))["numeric_summary"].items())[:2]})
Xc = DataCleaner().fit_transform(X)
print("cleaned", Xc.shape)

print("== 3) AutoML + Explain + Persist ==")
from salearn.automl import find_best
from salearn.explain import model_report
from salearn.persist import save_model, load_model
best, board = find_best(Xc, y, cv=2, time_budget=20)
print("best:", board[0]["model"], round(board[0]["mean"], 4))
print(model_report(best, Xc, y))
save_model(best, "/tmp/salearn_demo.ssl")
print("reload ok:", load_model("/tmp/salearn_demo.ssl").score(Xc, y))

print("== 4) Time-series ==")
from salearn.timeseries import ARForecaster
yt = np.arange(50, dtype=float) + np.sin(np.arange(50))
print("forecast:", ARForecaster(lags=5).fit(yt[:40]).predict(5).round(2))

print("== 5) Arabic NLP ==")
from salearn.text import TextClassifier
clf = TextClassifier(max_iter=100).fit(["فيلم رائع", "سيء وممل", "ممتع جدا", "ممل جدا"], [1, 0, 1, 0])
print("pred:", clf.predict(["رائع وممتع", "سيء"]))
print("OK — salearn full stack works.")
