"""Validate the alignment and finite static signals of the new specialist."""

import numpy as np
import pandas as pd
import pytest

from phishing_url.v13_features import extend_frame


def test_extra_signals_preserve_rows_and_do_not_use_categorical_domain_strings():
    base = pd.DataFrame({"baseline_probability": [.2, .8]})
    extra = pd.DataFrame({"extra_signal_0": [0, 2], "cat_extra_canonical_domain": ["a.example", "b.example"]})
    X = extend_frame(base, extra)
    assert list(X.columns) == ["baseline_probability", "static_extra_signal_0"]
    np.testing.assert_array_equal(X.to_numpy(), [[.2, 0], [.8, 2]])


def test_extra_rows_and_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="align"):
        extend_frame(pd.DataFrame({"p": [.2, .8]}), pd.DataFrame({"extra_signal": [1]}))
    with pytest.raises(ValueError, match="finite"):
        extend_frame(pd.DataFrame({"p": [.2]}), pd.DataFrame({"extra_signal": [float("nan")]}))
