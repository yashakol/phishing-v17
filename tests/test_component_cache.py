import pytest

from phishing_url.content_checkpoints import guard, load_component, save_component


def test_full_fit_cannot_be_reused_as_holdout_training_model(tmp_path):
    cache = {"urls": ["https://example.com/", "https://example.org/"], "labels": [0, 1]}
    full = guard(cache, [0, 1], {"iterations": 100}, "same-data")
    dev = guard(cache, [0], {"iterations": 100}, "same-data")
    path = tmp_path / "component.joblib"
    save_component(path, {"test_model": "full-data"}, full)
    with pytest.raises(ValueError, match="scope differs"):
        load_component(path, dev)
    assert load_component(path, full)["test_model"] == "full-data"
