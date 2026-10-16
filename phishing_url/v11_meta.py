"""Learn a small ensemble from aligned out-of-sample component predictions."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

COMPONENTS = ("svm", "c16", "numeric", "url", "cat", "hashed", "extra")
STRUCTURAL = ("html_text_length", "html_title_length", "html_tag_form", "html_input_password",
              "html_form_external_ratio", "html_links_external_ratio", "html_scripts_inline",
              "html_empty", "url_length", "host_length")


def ensemble_frame(components, structural, routed, counts, prior):
    if set(components) != set(COMPONENTS):
        raise ValueError("Component schema differs")
    n = len(structural)
    output, probabilities = {}, []
    for name in COMPONENTS:
        p = np.asarray(components[name], dtype=float)
        if p.shape != (n,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Component probabilities must be aligned and finite")
        output[name] = p
        output["logit_"+name] = logit(np.clip(p, 1e-6, 1-1e-6))
        probabilities.append(p)
    p = np.vstack(probabilities)
    output.update(probability_minimum=p.min(axis=0), probability_maximum=p.max(axis=0),
                  probability_std=p.std(axis=0), positive_votes=(p >= .5).sum(axis=0))
    routed = np.asarray(routed)
    if routed.shape != (n,) or not np.isin(routed, [False, True]).all():
        raise ValueError("Routing mask must align")
    output["url_routed"] = routed.astype(float)
    for name in STRUCTURAL:
        if name not in structural:
            raise ValueError("Structural ensemble schema differs")
        values = structural[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Structural features must be finite and nonnegative")
        output[name] = values if name.endswith("ratio") else np.log1p(values)
    prior = np.broadcast_to(np.asarray(prior, dtype=float), (n,))
    if not np.isfinite(prior).all() or (prior <= 0).any() or (prior >= 1).any():
        raise ValueError("Training prior must be strictly between zero and one")
    for kind in ("domain", "title"):
        values = np.asarray(counts[kind])
        if values.shape != (n, 2) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Training-only evidence counts must align")
        negative, positive = values.T
        total = negative+positive
        posterior = (positive+5.*prior)/(total+5.)
        output[kind+"_log_support"] = np.log1p(total)
        output[kind+"_log_evidence"] = logit(posterior)-logit(prior)
        output[kind+"_pure"] = ((total > 0) & ((negative == 0) | (positive == 0))).astype(float)
    return pd.DataFrame(output)


class LearnedEnsemble(BaseEstimator):
    def __init__(self, kind="linear", c=.1, depth=3, iterations=400, threads=2,
                 random_state=20261010):
        self.kind, self.c, self.depth = kind, c, depth
        self.iterations, self.threads, self.random_state = iterations, threads, random_state

    def fit(self, frame, labels):
        y = np.asarray(labels)
        if len(frame) != len(y) or not np.array_equal(np.unique(y), [0, 1]) or \
                not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Ensemble training rows and binary labels must align")
        if self.threads not in (1, 2):
            raise ValueError("Threads must be bounded")
        self.columns_ = list(frame.columns)
        if self.kind == "linear":
            self.model_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.c, solver="lbfgs",
                max_iter=1200, random_state=self.random_state))
        elif self.kind == "tree":
            self.model_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth,
                learning_rate=.04, l2_leaf_reg=30., loss_function="Logloss", random_seed=self.random_state,
                thread_count=self.threads, verbose=False, allow_writing_files=False)
        else:
            raise ValueError("Unknown ensemble classifier kind")
        self.model_.fit(frame, y)
        self.training_rows_ = len(y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, frame):
        check_is_fitted(self, ["model_", "columns_"])
        if list(frame.columns) != self.columns_ or not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Ensemble feature schema differs")
        return self.model_.predict_proba(frame)[:, 1]
