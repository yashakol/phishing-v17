import numpy as np
import pandas as pd
import pytest

from phishing_url.v16_expert import CATEGORY_KEYS, HTML_KEYS, URL_KEYS, category_frame, make_specialist, mix_specialist, specialist_frame
from phishing_url.v15_resources import KEYS


def fixture_rows():
    cache = {"urls": ["https://a.example/"], "records": [{"url": "https://a.example/", "title": "  SIGN   IN  "}],
             "features": pd.DataFrame({key: ["en"] for key in HTML_KEYS})}
    extra = {"urls": cache["urls"], "features": pd.DataFrame({key: ["EMPTY"] for key in KEYS})}
    url = pd.DataFrame({key: ["a"] for key in URL_KEYS})
    return cache, extra, url


def test_categories_normalize_title_and_preserve_expected_schema():
    result = category_frame(*fixture_rows())
    assert result.columns.tolist() == list(CATEGORY_KEYS)
    assert result.cat_normalized_title.iloc[0] == "sign in"


def test_categories_reject_reordered_records():
    cache, extra, url = fixture_rows()
    cache["records"][0]["url"] = "https://b.example/"
    with pytest.raises(ValueError, match="order"):
        category_frame(cache, extra, url)


def test_categories_reject_missing_or_nonstring_values():
    cache, extra, url = fixture_rows()
    extra["features"].iloc[0, 0] = None
    with pytest.raises(ValueError, match="strings"):
        category_frame(cache, extra, url)


def test_categorical_specialist_carries_strings_without_numeric_casting():
    cats = category_frame(*fixture_rows())
    out = specialist_frame(pd.DataFrame({"x": [1.]}), cats, "categorical")
    assert len(out.columns) == 13 and out.cat_normalized_title.iloc[0] == "sign in"
    assert specialist_frame(pd.DataFrame({"x": [1.]}), cats, "numeric_deep").columns.tolist() == ["x"]


def test_numeric_specialist_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="finite"):
        specialist_frame(pd.DataFrame({"x": [np.inf]}), category_frame(*fixture_rows()), "numeric_deep")


def test_new_mixture_preserves_frozen_confident_and_hard_outputs():
    p = np.array([0., .03, .2, .7, .99, 1.])
    v15 = np.array([0., .03, .1, .8, .99, 1.])
    q = np.array([1., 1., 1., 0., 0., 0.])
    np.testing.assert_array_equal(mix_specialist(p, v15, q, .5), [0., .03, .55, .4, .99, 1.])


def test_new_mixture_rejects_misaligned_probabilities():
    with pytest.raises(ValueError, match="align"):
        mix_specialist([.2], [.2, .3], [.4], .5)


def test_specialist_factory_is_bounded_and_rejects_unknown_kind():
    assert make_specialist("categorical").get_params()["cat_features"] == list(CATEGORY_KEYS)
    assert make_specialist("numeric_deep").get_params()["thread_count"] == 2
    with pytest.raises(ValueError, match="Unknown"):
        make_specialist("other")
