"""Training-only nearest-page evidence, using bounded inert HTML text."""

import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from sparse_dot_topn import sp_matmul_topn


def adjust_neighbors(base, similarities, labels, config):
    base = np.asarray(base, dtype=float)
    similarities = np.asarray(similarities)
    labels = np.asarray(labels)
    if similarities.shape != labels.shape or len(base) != len(labels):
        raise ValueError("Neighbor arrays and probabilities must align")
    count = config.get("neighbors", similarities.shape[1])
    scores = similarities[:, :count]
    y = labels[:, :count]
    valid = (y >= 0) & (scores >= config["minimum_similarity"])
    weights = np.where(valid, np.maximum(scores, 0) ** config.get("power", 4), 0.)
    total = weights.sum(axis=1)
    prior = config.get("prior", .375)
    smoothing = config.get("smoothing", .5)
    probability = ((weights * np.maximum(y, 0)).sum(axis=1) + smoothing * prior) / (total + smoothing)
    top = np.max(scores, axis=1)
    influence = config["weight"] * np.clip(
        (top - config["minimum_similarity"]) / (1 - config["minimum_similarity"]), 0, 1)
    eligible = (valid.sum(axis=1) >= config.get("minimum_support", 2)) & (base > 0) & (base < 1)
    result = base.copy()
    result[eligible] = (1 - influence[eligible]) * base[eligible] + influence[eligible] * probability[eligible]
    return result


class NeighborModel(BaseEstimator):
    def __init__(self, view="combined", max_features=180000, max_neighbors=20, threads=2):
        self.view = view
        self.max_features = max_features
        self.max_neighbors = max_neighbors
        self.threads = threads

    def documents(self, records):
        fields = {}
        if self.view in {"content", "combined"}:
            fields["content"] = [((r.get("title", "") + " ") * 3 + r.get("text", ""))[:13803] for r in records]
        if self.view in {"markup", "combined"}:
            fields["markup"] = [r.get("markup", "")[:16000] for r in records]
        if self.view == "url":
            fields["url"] = [r["url"][:4096] for r in records]
        if not fields:
            raise ValueError("Unknown neighbor representation")
        return fields

    def matrix(self, records, fit=False):
        fields = self.documents(records)
        matrices = []
        if fit:
            self.vectorizers_ = {}
        for name, docs in fields.items():
            if fit:
                params = dict(min_df=2, max_df=.95, max_features=self.max_features,
                              sublinear_tf=True, dtype=np.float32, strip_accents="unicode")
                if name == "url":
                    params.update(analyzer="char", ngram_range=(3, 5))
                else:
                    params.update(ngram_range=(1, 2), token_pattern=r"(?u)[\w:-]+")
                vectorizer = TfidfVectorizer(**params)
                try:
                    current = vectorizer.fit_transform(docs)
                except ValueError as exc:
                    if "empty vocabulary" in str(exc) or "After pruning" in str(exc):
                        continue
                    raise
                self.vectorizers_[name] = vectorizer
            elif name in self.vectorizers_:
                current = self.vectorizers_[name].transform(docs)
            else:
                continue
            matrices.append(current)
        if not matrices:
            raise ValueError("No usable neighbor vocabulary")
        return normalize(hstack(matrices, format="csr"), norm="l2", copy=False)

    def fit(self, records, y):
        records = list(records)
        self.labels_ = np.asarray(y, dtype=np.int8)
        if len(records) != len(y) or not np.array_equal(np.unique(self.labels_), [0, 1]):
            raise ValueError("Neighbor training requires aligned binary labels")
        self.training_matrix_ = self.matrix(records, fit=True)
        self.training_rows_ = len(records)
        self.prior_ = float(self.labels_.mean())
        return self

    def lookup(self, records):
        check_is_fitted(self, "training_matrix_")
        records = list(records)
        query = self.matrix(records)
        count = min(self.max_neighbors, self.training_rows_)
        found = sp_matmul_topn(query, self.training_matrix_.T, top_n=count,
                               threshold=.25, sort=True, n_threads=self.threads)
        similarities = np.zeros((len(records), self.max_neighbors), dtype=np.float32)
        labels = np.full((len(records), self.max_neighbors), -1, dtype=np.int8)
        for i in range(len(records)):
            start, end = found.indptr[i:i + 2]
            scores, indices = found.data[start:end], found.indices[start:end]
            n = len(scores)
            similarities[i, :n] = scores
            labels[i, :n] = self.labels_[indices]
        return similarities, labels
