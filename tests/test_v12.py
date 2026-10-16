"""Contracts for a specialist restricted to uncertain frozen predictions."""

import numpy as np
import pandas as pd
import pytest

from phishing_url.v12_expert import blend_expert, eligibility, expert_frame
from phishing_url.v12_bundle import PreparedProbability


def test_expert_can_only_change_eligible_rows_in_original_order():
    p = np.array([0, .01, .05, .2, .5, .8, .95, .99, 1])
    q = 1-p
    after = blend_expert(p, q, .75)
    eligible = eligibility(p)
    np.testing.assert_array_equal(after[~eligible], p[~eligible])
    np.testing.assert_allclose(after[eligible], [.725, .65, .5, .35, .275])
    np.testing.assert_array_equal(p, [0, .01, .05, .2, .5, .8, .95, .99, 1])


def test_frame_excludes_categorical_strings_but_keeps_numeric_column_order():
    meta = pd.DataFrame({"p": [.2, .8]})
    raw = pd.DataFrame({"cat_scheme": ["https", "http"], "length": [12, 20], "https": [True, False]})
    X = expert_frame(meta, raw, [.2, .8])
    assert list(X.columns) == ["p", "numeric_length", "numeric_https", "baseline_probability", "baseline_logit"]
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_frame_and_probability_alignment_errors_are_rejected():
    with pytest.raises(ValueError, match="aligned"):
        expert_frame(pd.DataFrame({"p": [.2]}), pd.DataFrame({"length": [1, 2]}), [.2])
    with pytest.raises(ValueError, match="align"):
        blend_expert([.2, .8], [.1], .5)


@pytest.mark.parametrize("weight", [float("nan"), -.1, 1.1])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError, match="weight"):
        blend_expert([.2], [.8], weight)


@pytest.mark.parametrize("baseline", [[float("nan")], [-.1], [1.1]])
def test_invalid_baseline_probabilities_rejected(baseline):
    with pytest.raises(ValueError, match="Baseline"):
        eligibility(baseline)


def test_prepared_component_rejects_reordered_records():
    records = [{"url": "https://first.example/"}, {"url": "https://second.example/"}]
    component = PreparedProbability(records, [.2, .8])
    np.testing.assert_array_equal(component.predict_proba(records), [.2, .8])
    with pytest.raises(ValueError, match="order"):
        component.predict_proba(records[::-1])
