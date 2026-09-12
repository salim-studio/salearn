"""salearn — from SQL to serving in one import.

A sklearn-compatible, faster ML library (NumPy + optional Numba) with a
built-in data stack: databases, loading, EDA, cleaning, feature engineering,
time-series forecasting, NLP, AutoML, explainability and model registry.

    # sklearn-style, drop-in
    from salearn.ensemble import RandomForestClassifier
    # full stack
    from salearn.db import Database
    from salearn import eda, io, clean, features, timeseries, text, automl, explain, persist, viz
"""
from __future__ import annotations

__version__ = "1.0.0"

from . import (
    base, utils, exceptions, preprocessing, impute, linear_model, tree,
    ensemble, neighbors, naive_bayes, svm, cluster, decomposition, metrics,
    model_selection, pipeline, feature_selection, mixture, neural_network,
    extras, datasets, calibration,
    db, io, eda, clean, features, timeseries, text, automl, explain,
    persist, viz,
)
from .base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, ClusterMixin, clone, is_classifier, is_regressor
from .pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from .model_selection import (
    train_test_split, KFold, StratifiedKFold, ShuffleSplit, LeaveOneOut,
    cross_val_score, cross_validate, cross_val_predict, GridSearchCV,
    RandomizedSearchCV, learning_curve, validation_curve,
)
from . import metrics as metrics_module

__all__ = [
    "__version__",
    "base", "utils", "exceptions", "preprocessing", "impute", "linear_model",
    "tree", "ensemble", "neighbors", "naive_bayes", "svm", "cluster",
    "decomposition", "metrics", "model_selection", "pipeline",
    "feature_selection", "mixture", "neural_network", "extras", "datasets",
    "calibration",
    "db", "io", "eda", "clean", "features", "timeseries", "text",
    "automl", "explain", "persist", "viz",
    "BaseEstimator", "ClassifierMixin", "RegressorMixin", "TransformerMixin",
    "ClusterMixin", "clone", "is_classifier", "is_regressor",
    "Pipeline", "FeatureUnion", "make_pipeline", "make_union",
    "train_test_split", "KFold", "StratifiedKFold", "ShuffleSplit",
    "LeaveOneOut", "cross_val_score", "cross_validate", "cross_val_predict",
    "GridSearchCV", "RandomizedSearchCV", "learning_curve", "validation_curve",
]
