"""Two complementary estimators: text patterns and structural gradient boosting."""

import time

import numpy as np
from catboost import CatBoostClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .features import categorical_columns, text_tokens


class TextModel:
    def __init__(self, kind="tfidf", c=4.0):
        self.kind = kind
        self.c = c
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                   max_features=300000, sublinear_tf=True, dtype=np.float32)
        self.word = TfidfVectorizer(analyzer=text_tokens, min_df=2, max_features=100000,
                                   sublinear_tf=True, dtype=np.float32)
        self.model = LogisticRegression(C=c, solver="liblinear", max_iter=600, random_state=2026)
        self.ratio = None

    def fit(self, urls, labels):
        started = time.perf_counter()
        matrix = hstack([self.char.fit_transform(urls), self.word.fit_transform(urls)], format="csr")
        if self.kind == "nb":
            positive = np.asarray(matrix[labels == 1].sum(axis=0)).ravel() + 1.0
            negative = np.asarray(matrix[labels == 0].sum(axis=0)).ravel() + 1.0
            positive /= positive.sum()
            negative /= negative.sum()
            self.ratio = np.log(positive / negative)
            matrix = matrix.multiply(self.ratio).tocsr()
        self.model.fit(matrix, labels)
        self.fit_seconds = time.perf_counter() - started
        self.n_features = matrix.shape[1]
        return self

    def predict_proba(self, urls):
        matrix = hstack([self.char.transform(urls), self.word.transform(urls)], format="csr")
        if self.ratio is not None:
            matrix = matrix.multiply(self.ratio).tocsr()
        return self.model.predict_proba(matrix)[:, 1]


def fit_boost(frame, labels, seed=2026, iterations=1000, threads=4):
    started = time.perf_counter()
    model = CatBoostClassifier(iterations=iterations, depth=7, learning_rate=0.055,
                              l2_leaf_reg=5, loss_function="Logloss", random_seed=seed,
                              thread_count=threads, verbose=False, allow_writing_files=False)
    model.fit(frame, labels, cat_features=categorical_columns(frame))
    model.fit_seconds = time.perf_counter() - started
    return model


def boost_probability(model, frame):
    return model.predict_proba(frame)[:, 1]
