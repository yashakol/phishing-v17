"""Offline sparse classifier over already-extracted static HTML records.

This module treats every field as inert text. It does not fetch URLs, parse or
execute scripts, or consult reference datasets. Fit vocabularies on training
records only. The returned probability vector is P(label=1).
"""

import time

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


class ContentTextModel(BaseEstimator):
    """Bounded TF-IDF content, markup, and optional URL views + logistic model.

    ``nb=True`` learns a smoothed supervised log-count ratio on the fit split.
    Character views use only the leading ``char_text_chars`` visible characters
    to bound vocabulary construction cost. Full word features use 12000 chars.
    """

    def __init__(self, c=4.0, nb=False, include_url=True, include_char=True,
                 include_markup=True, min_df=2, word_features=140000,
                 char_features=100000, markup_features=70000,
                 url_features=90000, char_text_chars=2000, max_iter=500,
                 threads=2, random_state=20261006, vocabulary_fit_rows=10000,
                 transform_batch_size=2048, verbose=False):
        self.c = c
        self.nb = nb
        self.include_url = include_url
        self.include_char = include_char
        self.include_markup = include_markup
        self.min_df = min_df
        self.word_features = word_features
        self.char_features = char_features
        self.markup_features = markup_features
        self.url_features = url_features
        self.char_text_chars = char_text_chars
        self.max_iter = max_iter
        self.threads = threads
        self.random_state = random_state
        self.vocabulary_fit_rows = vocabulary_fit_rows
        self.transform_batch_size = transform_batch_size
        self.verbose = verbose

    def _documents(self, records):
        fields = {key: [] for key in ("word", "char", "markup", "url")}
        for record in records:
            def bounded(key, limit):
                value = record.get(key, "")
                if value is None:
                    return ""
                if not isinstance(value, str):
                    raise TypeError(f"record[{key!r}] must be a string")
                return value[:limit]
            title = bounded("title", 600)
            text = bounded("text", 12000)
            fields["word"].append(title + " " + title + " " + text)
            fields["char"].append(title + " " + text[:self.char_text_chars])
            fields["markup"].append(bounded("markup", 16000))
            fields["url"].append(bounded("url", 4096))
        return fields

    def _specifications(self):
        specs = [("word", dict(analyzer="word", ngram_range=(1, 2),
                               token_pattern=r"(?u)\b\w\w+\b",
                               max_features=self.word_features))]
        if self.include_char:
            specs.append(("char", dict(analyzer="char_wb", ngram_range=(3, 5),
                                       max_features=self.char_features)))
        if self.include_markup:
            specs.append(("markup", dict(analyzer="word", ngram_range=(1, 2),
                                         token_pattern=r"(?u)[\w:-]+",
                                         max_features=self.markup_features)))
        if self.include_url:
            specs.append(("url", dict(analyzer="char", ngram_range=(3, 5),
                                      max_features=self.url_features)))
        return specs

    def _matrix(self, records, fit=False):
        documents = self._documents(records)
        matrices = []
        if fit:
            self.vectorizers_ = {}
            self.view_fit_seconds_ = {}
            if len(records) > self.vocabulary_fit_rows:
                vocabulary_rows = np.sort(np.random.RandomState(self.random_state).choice(
                    len(records), self.vocabulary_fit_rows, replace=False))
            else:
                vocabulary_rows = np.arange(len(records))
            self.vocabulary_fit_rows_ = len(vocabulary_rows)
            for name, parameters in self._specifications():
                started = time.perf_counter()
                vectorizer = TfidfVectorizer(min_df=self.min_df,
                                             sublinear_tf=True, dtype=np.float32,
                                             strip_accents="unicode", **parameters)
                try:
                    if len(vocabulary_rows) == len(records):
                        matrix = vectorizer.fit_transform(documents[name])
                    else:
                        vectorizer.fit([documents[name][i] for i in vocabulary_rows])
                        matrix = self._transform_batched(vectorizer, documents[name])
                except ValueError as error:
                    # An empty HTML field is valid; it contributes no features.
                    if not any(s in str(error) for s in
                               ("empty vocabulary", "After pruning", "max_df corresponds")):
                        raise
                    continue
                self.vectorizers_[name] = vectorizer
                matrices.append(matrix)
                self.view_fit_seconds_[name] = time.perf_counter() - started
                if self.verbose:
                    print(f"content_view={name} features={matrix.shape[1]} nnz={matrix.nnz} "
                          f"seconds={self.view_fit_seconds_[name]:.2f}", flush=True)
        else:
            for name, vectorizer in self.vectorizers_.items():
                matrices.append(self._transform_batched(vectorizer, documents[name]))
        if not matrices:
            return csr_matrix(np.ones((len(records), 1), dtype=np.float32))
        return hstack(matrices, format="csr", dtype=np.float32)

    def _transform_batched(self, vectorizer, documents):
        parts = [vectorizer.transform(documents[i:i + self.transform_batch_size])
                 for i in range(0, len(documents), self.transform_batch_size)]
        return vstack(parts, format="csr") if len(parts) > 1 else parts[0]

    def fit(self, records, y):
        records = list(records)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(records) != len(labels) or len(records) < 2:
            raise ValueError("records and one-dimensional labels must align (at least 2 rows)")
        if not np.array_equal(np.unique(labels), [0, 1]):
            raise ValueError("both binary classes 0 and 1 are required")
        if not isinstance(self.threads, int) or not 1 <= self.threads <= 2:
            raise ValueError("threads must be 1 or 2")
        if not isinstance(self.char_text_chars, int) or not 0 <= self.char_text_chars <= 12000:
            raise ValueError("char_text_chars must be between 0 and 12000")
        if (not isinstance(self.vocabulary_fit_rows, int) or self.vocabulary_fit_rows < 2 or
                not isinstance(self.transform_batch_size, int) or self.transform_batch_size < 1):
            raise ValueError("vocabulary_fit_rows must be >=2 and transform_batch_size >=1")
        started = time.perf_counter()
        with threadpool_limits(limits=self.threads):
            matrix = self._matrix(records, fit=True)
            self.ratio_ = None
            if self.nb:
                positive = np.asarray(matrix[labels == 1].sum(axis=0)).ravel() + 1
                negative = np.asarray(matrix[labels == 0].sum(axis=0)).ravel() + 1
                self.ratio_ = np.log((positive / positive.sum()) /
                                     (negative / negative.sum())).astype(np.float32)
                matrix = matrix.multiply(self.ratio_).tocsr()
            self.classifier_ = LogisticRegression(C=self.c, solver="liblinear",
                                                  max_iter=self.max_iter,
                                                  random_state=self.random_state)
            self.classifier_.fit(matrix, labels)
        self.classes_ = self.classifier_.classes_
        self.n_features_ = matrix.shape[1]
        self.n_features = self.n_features_
        self.fit_seconds_ = time.perf_counter() - started
        self.fit_seconds = self.fit_seconds_
        return self

    def predict_proba(self, records):
        check_is_fitted(self, "classifier_")
        records = list(records)
        if not records:
            return np.empty(0, dtype=float)
        with threadpool_limits(limits=self.threads):
            matrix = self._matrix(records)
            if self.ratio_ is not None:
                matrix = matrix.multiply(self.ratio_).tocsr()
            return self.classifier_.predict_proba(matrix)[:, 1]
