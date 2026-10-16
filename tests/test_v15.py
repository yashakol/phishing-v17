"""Ensure resource context cannot encode a training row's own resource label."""

import numpy as np
import pandas as pd
import pytest

from phishing_url.v15_resources import KEYS, ResourceContext


def feature(keys):
    return pd.DataFrame({kind: keys for kind in KEYS})


def test_unique_training_resource_has_neutral_evidence_when_own_label_is_left_out():
    X = feature(["safe.example", "phish.example"])
    context = ResourceContext().fit(X, [0, 1])
    after = context.transform(X, [0, 1])
    np.testing.assert_array_equal(after.to_numpy(), 0)
    assert context.transform(X).to_numpy().any()
    # Inference does not change the stored training counters.
    assert context.counts_[KEYS[0]]["phish.example"] == [0, 1]


def test_unseen_and_missing_resources_are_neutral():
    context = ResourceContext().fit(feature(["a.example", "b.example"]), [0, 1])
    np.testing.assert_array_equal(context.transform(feature(["unknown.example", "EMPTY", "INVALID"])).to_numpy(), 0)


def test_leave_one_out_rejects_rows_without_corresponding_training_label():
    context = ResourceContext().fit(feature(["a.example", "b.example"]), [0, 1])
    with pytest.raises(ValueError, match="belong"):
        context.transform(feature(["a.example"]), [1])
