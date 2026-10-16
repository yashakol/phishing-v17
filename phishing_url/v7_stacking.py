"""A small classifier learning when to trust page text and structure.

Training requires honest out-of-sample base predictions. The caller supplies
those predictions; this module does not load any labels or external datasets.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STRUCTURAL = ("html_text_length", "html_title_length", "html_tag_form", "html_input_password",
              "html_input_password_ratio", "html_form_external_ratio", "html_links_external_ratio",
              "html_scripts_inline", "url_length", "host_length")


def stacking_frame(text, numeric, structural):
    text, numeric = np.asarray(text, dtype=float), np.asarray(numeric, dtype=float)
    if text.shape != numeric.shape or text.shape != (len(structural),) or \
            not np.isfinite(text).all() or not np.isfinite(numeric).all() or \
            (text < 0).any() or (text > 1).any() or (numeric < 0).any() or (numeric > 1).any():
        raise ValueError("Stacking probabilities must align and be finite in [0,1]")
    logit = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6))
    output = {"text": text, "numeric": numeric, "logit_text": logit(text), "logit_numeric": logit(numeric),
              "difference": text - numeric, "disagreement": abs(text - numeric),
              "product": text * numeric, "minimum": np.minimum(text, numeric),
              "maximum": np.maximum(text, numeric)}
    for column in STRUCTURAL:
        if column in structural:
            values = structural[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid structural feature for stacking")
            output[column] = np.log1p(values) if not column.endswith("ratio") else values
    return pd.DataFrame(output)


class PageStacker:
    def __init__(self, kind="linear", c=1., depth=3, iterations=400, threads=2):
        self.kind, self.c = kind, float(c)
        self.depth, self.iterations, self.threads = int(depth), int(iterations), int(threads)

    def fit(self, frame, labels):
        self.columns_ = list(frame.columns)
        if self.kind == "linear":
            self.model_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.c, solver="lbfgs",
                max_iter=1500, random_state=20261008))
        elif self.kind == "tree":
            self.model_ = CatBoostClassifier(depth=self.depth, iterations=self.iterations,
                learning_rate=.035, l2_leaf_reg=8., loss_function="Logloss", random_seed=20261008,
                thread_count=self.threads, verbose=False, allow_writing_files=False)
        else:
            raise ValueError("Unknown stacker kind")
        self.model_.fit(frame, labels)
        self.training_rows_ = len(frame)
        return self

    def predict_proba(self, frame):
        if list(frame.columns) != self.columns_:
            raise ValueError("Stacking feature order differs")
        return self.model_.predict_proba(frame)[:, 1]
