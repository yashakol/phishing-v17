"""Smoothed resource-domain evidence fitted exclusively on allowed training rows."""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import logit

KEYS = ("cat_extra_canonical_domain", "cat_extra_og_domain", "cat_extra_main_resource_domain")
MISSING = {"", "EMPTY", "INVALID"}


class ResourceContext:
    def fit(self, features, labels):
        y = np.asarray(labels)
        if len(features) != len(y) or not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Resource training rows and binary labels must align")
        self.prior_ = float(y.mean())
        self.training_rows_ = len(y)
        self.counts_ = {}
        for kind in KEYS:
            counts = defaultdict(lambda: [0, 0])
            for key, label in zip(features[kind].tolist(), y, strict=True):
                if key not in MISSING:
                    counts[key][int(label)] += 1
            self.counts_[kind] = dict(counts)
        return self

    def transform(self, features, leave_out_labels=None):
        output = {}
        if leave_out_labels is not None:
            labels = np.asarray(leave_out_labels)
            if labels.shape != (len(features),) or not np.isin(labels, [0, 1]).all():
                raise ValueError("Leave-one-out labels must align")
        for kind in KEYS:
            counts = np.asarray([self.counts_[kind].get(key, [0, 0]) for key in features[kind]], dtype=float)
            if len(counts) == 0:
                counts = np.zeros((0, 2))
            if leave_out_labels is not None:
                active = np.flatnonzero(~features[kind].isin(MISSING).to_numpy())
                counts[active, labels[active].astype(int)] -= 1
                if (counts < 0).any():
                    raise ValueError("Leave-one-out rows must belong to the fitted context")
            zero, one = counts.T
            total = zero+one
            posterior = (one+10*self.prior_)/(total+10)
            output[kind+"_log_support"] = np.log1p(total)
            output[kind+"_log_evidence"] = logit(posterior)-logit(self.prior_)
            output[kind+"_pure"] = ((total > 0) & ((zero == 0) | (one == 0))).astype(float)
        return pd.DataFrame(output)


def add_resource_evidence(base, features, context, leave_out_labels=None):
    if len(base) != len(features):
        raise ValueError("Resource evidence rows must align")
    result = pd.concat([base.reset_index(drop=True), context.transform(features, leave_out_labels)], axis=1)
    if result.columns.duplicated().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Resource evidence schema must be unique and finite")
    return result
