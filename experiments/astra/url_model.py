"""Offline URL-only sparse models. All fitted statistics use the fit partition."""
import re
import time
from urllib.parse import unquote

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from phishing_url.features import split_url, domain_parts, text_tokens


def path_template(text):
    text = unquote(text).lower()
    text = re.sub(r"[0-9a-f]{16,}", "HEX", text)
    text = re.sub(r"[0-9]+", "N", text)
    return text


def host_view(url):
    return "^" + (split_url(url).hostname or "").lower() + "$"


def tail_view(url):
    p = split_url(url)
    return "^" + unquote(p.path).lower() + ("?" + unquote(p.query).lower() if p.query else "") + "$"


def rich_tokens(url):
    p = split_url(url)
    host = (p.hostname or "").lower()
    domain, suffix, subdomain, stem, _ = domain_parts(host)
    path = unquote(p.path).lower()
    segments = [s for s in path.split("/") if s]
    norm = [path_template(s) for s in segments]
    tokens = text_tokens(url)
    tokens += ["stem=" + stem, "subdomain=" + subdomain,
               "path_template=" + "/".join(norm),
               "domain_depth=" + domain + ":" + str(min(len(segments), 8))]
    if norm:
        tokens += ["first_template=" + norm[0], "last_template=" + norm[-1],
                   "domain_first=" + domain + ":" + norm[0],
                   "suffix_first=" + suffix + ":" + norm[0]]
    for i, s in enumerate(norm):
        tokens += ["segment=" + s, "position_segment=" + str(min(i, 4)) + ":" + s]
    tokens += ["segment_pair=" + a + "/" + b for a, b in zip(norm, norm[1:])]
    words = re.findall(r"[a-z0-9]+", path)
    tokens += ["path_pair=" + a + ":" + b for a, b in zip(words, words[1:])]
    keys = sorted(k.split("=", 1)[0].lower() for k in p.query.split("&") if k)
    tokens += ["query_key=" + k for k in keys]
    tokens += ["query_keys=" + ":".join(keys), "path_query=" + "/".join(norm) + "?" + ":".join(keys)]
    labels = host.split(".")
    tokens += ["host_pair=" + a + "." + b for a, b in zip(labels, labels[1:])]
    return tokens


class URLFeatures:
    def __init__(self):
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                   max_features=300000, sublinear_tf=True, dtype=np.float32)
        self.base_word = TfidfVectorizer(analyzer=text_tokens, min_df=2, max_features=100000,
                                        sublinear_tf=True, dtype=np.float32)
        self.rich_word = TfidfVectorizer(analyzer=rich_tokens, min_df=2, max_features=200000,
                                        sublinear_tf=True, dtype=np.float32)
        self.host = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                   max_features=160000, sublinear_tf=True, dtype=np.float32)
        self.tail = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), min_df=2,
                                   max_features=350000, sublinear_tf=True, dtype=np.float32)

    def matrices(self, urls, fit=False, only=None):
        operation = "fit_transform" if fit else "transform"
        c = getattr(self.char, operation)(urls)
        result = {}
        if only in (None, "base"):
            w = getattr(self.base_word, operation)(urls)
            result["base"] = hstack([c, w], format="csr")
        if only in (None, "rich"):
            w = getattr(self.rich_word, operation)(urls)
            h = getattr(self.host, operation)([host_view(u) for u in urls])
            t = getattr(self.tail, operation)([tail_view(u) for u in urls])
            result["rich"] = hstack([c, w, h * 0.7, t * 0.7], format="csr")
        return result


class SparseClassifier:
    def __init__(self, c=4, nb=False):
        self.c, self.nb = c, nb
        self.ratio = None
        self.model = LogisticRegression(C=c, solver="liblinear", max_iter=600,
                                        random_state=2026)

    def fit(self, matrix, y):
        start = time.perf_counter()
        if self.nb:
            pos = np.asarray(matrix[y == 1].sum(axis=0)).ravel() + 1
            neg = np.asarray(matrix[y == 0].sum(axis=0)).ravel() + 1
            self.ratio = np.log((pos / pos.sum()) / (neg / neg.sum())).astype(np.float32)
        self.model.fit(self.transform(matrix), y)
        self.fit_seconds = time.perf_counter() - start
        return self

    def transform(self, matrix):
        return matrix if self.ratio is None else matrix.multiply(self.ratio).tocsr()

    def predict_proba(self, matrix):
        return self.model.predict_proba(self.transform(matrix))[:, 1]


class URLModel:
    def __init__(self, representation="rich", c=4, nb=False):
        self.representation = representation
        self.features = URLFeatures()
        self.classifier = SparseClassifier(c=c, nb=nb)

    def fit(self, urls, y):
        start = time.perf_counter()
        x = self.features.matrices(urls, fit=True, only=self.representation)[self.representation]
        self.n_features = x.shape[1]
        self.classifier.fit(x, y)
        self.fit_seconds = time.perf_counter() - start
        return self

    def predict_proba(self, urls):
        x = self.features.matrices(urls, only=self.representation)[self.representation]
        return self.classifier.predict_proba(x)
