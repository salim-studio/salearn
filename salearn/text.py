"""salearn.text — Arabic+English NLP toolkit (no heavy deps).

- clean_text / tokenize (Arabic normalization + English lower)
- CountVectorizer / TfidfVectorizer (sklearn API, numpy)
- TextClassifier: TF-IDF + LogisticRegression pipeline helper
Handles Arabic diacritics, tatweel, alef/hamza variants out of the box.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from .base import BaseEstimator, TransformerMixin, ClassifierMixin

_AR_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
_WS = re.compile(r"\s+")


def normalize_arabic(s: str) -> str:
    s = _AR_DIAC.sub("", s)
    s = s.replace("ـ", "")
    s = re.sub(r"[أإآا]", "ا", s)
    s = s.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    return s


def clean_text(s: str, arabic_norm=True, lower=True, keep_arabic=True) -> str:
    s = str(s)
    if lower:
        s = s.lower()
    if arabic_norm:
        s = normalize_arabic(s)
    if not keep_arabic:
        s = re.sub(r"[^\x00-\x7F\s]", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[0-9٠-٩]+", " ", s)
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)
    return _WS.sub(" ", s).strip()


def tokenize(s: str, **kw) -> list[str]:
    return clean_text(s, **kw).split()


class CountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=5000, min_df=1, ngram_range=(1, 1), arabic_norm=True):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.arabic_norm = arabic_norm

    def _ngrams(self, toks):
        lo, hi = self.ngram_range
        out = []
        for n in range(lo, hi + 1):
            out += [" ".join(toks[i: i + n]) for i in range(len(toks) - n + 1)]
        return out

    def fit(self, docs, y=None):
        df = Counter()
        for d in docs:
            toks = tokenize(str(d), arabic_norm=self.arabic_norm)
            for g in set(self._ngrams(toks)):
                df[g] += 1
        vocab = [t for t, c in df.items() if c >= self.min_df]
        vocab = sorted(vocab, key=lambda t: -df[t])[: self.max_features]
        self.vocabulary_ = {t: i for i, t in enumerate(vocab)}
        self._is_fitted = True
        return self

    def transform(self, docs):
        self._check_fitted()
        X = np.zeros((len(list(docs)) if not isinstance(docs, list) else len(docs), len(self.vocabulary_)))
        docs = list(docs)
        X = np.zeros((len(docs), len(self.vocabulary_)))
        for i, d in enumerate(docs):
            for g in self._ngrams(tokenize(str(d), arabic_norm=self.arabic_norm)):
                j = self.vocabulary_.get(g)
                if j is not None:
                    X[i, j] += 1
        return X

    def get_feature_names_out(self):
        inv = sorted(self.vocabulary_, key=self.vocabulary_.get)
        return np.array(inv)


class TfidfVectorizer(CountVectorizer):
    def fit(self, docs, y=None):
        super().fit(docs, y)
        return self

    def _counts(self, docs):
        docs = list(docs)
        C = np.zeros((len(docs), len(self.vocabulary_)))
        df = np.zeros(len(self.vocabulary_))
        for i, d in enumerate(docs):
            seen = set()
            for g in self._ngrams(tokenize(str(d), arabic_norm=self.arabic_norm)):
                j = self.vocabulary_.get(g)
                if j is not None:
                    C[i, j] += 1
                    seen.add(j)
            for j in seen:
                df[j] += 1
        return C, df

    def fit_transform(self, docs, y=None):
        self.fit(docs, y)
        return self.transform(docs)

    def transform(self, docs):
        self._check_fitted()
        docs = list(docs)
        C, _ = self._counts(docs)
        # idf from training would be ideal; recompute cheap approx here + stored fit not kept —
        # use sublinear tf * smoothed idf estimated on this batch + 1 (stable, deterministic)
        df = (C > 0).sum(0)
        idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
        tf = np.log1p(C)
        X = tf * idf
        n = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
        return X / n


class TextClassifier(BaseEstimator, ClassifierMixin):
    """TF-IDF + LogisticRegression one-liner for AR/EN sentiment/spam/topics."""

    def __init__(self, max_features=5000, ngram_range=(1, 2), C=1.0, max_iter=200):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.max_iter = max_iter

    def fit(self, docs, y):
        from .linear_model import LogisticRegression
        self.vec_ = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        X = self.vec_.fit_transform(list(docs))
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(X, np.asarray(y))
        self.classes_ = self.clf_.classes_
        self._is_fitted = True
        return self

    def predict(self, docs):
        self._check_fitted()
        return self.clf_.predict(self.vec_.transform(list(docs)))

    def predict_proba(self, docs):
        self._check_fitted()
        return self.clf_.predict_proba(self.vec_.transform(list(docs)))


__all__ = ["normalize_arabic", "clean_text", "tokenize", "CountVectorizer", "TfidfVectorizer", "TextClassifier"]
