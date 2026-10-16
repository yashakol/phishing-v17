import numpy as np
import pytest

from phishing_url.content_refinement import (apply_values, fit_evidence, normalized_text,
                                            signature_keys)


def test_normalized_signature_ignores_dynamic_numbers_but_keeps_content():
    a = "Verify account 17372 https://sample.invalid/login?id=7"
    b = "Verify account 98461 https://other.invalid/a?id=19"
    assert normalized_text(a) == normalized_text(b)
    assert normalized_text(a) != normalized_text("News about account 17372")


def test_conflicting_training_keys_and_missing_text_are_not_evidence():
    memory = fit_evidence({"visible": ["a", "a", "b", "b", None]}, [0, 1, 1, 1, 0], {"visible": 2})
    assert memory == {"visible": {"b": 1}}


def test_evidence_only_uses_fitted_training_keys():
    memory = fit_evidence({"raw": ["train"]}, [1], {"raw": 1})
    assert "validation" not in memory["raw"]
    records = [{"url": "https://example.invalid/", "title": "", "text": "", "markup": ""}]
    keys = signature_keys(records, ["raw-hash"])
    assert keys["visible"] == [None]
    assert keys["normalized"] == [None]


def test_consensus_and_page_priority_resolve_weak_domain_disagreement():
    values = {"raw": np.array([1]), "domain": np.array([0])}
    assert apply_values([.4], values, "consensus")[0].tolist() == [.4]
    assert apply_values([.4], values, "page_priority")[0].tolist() == [1.]


def test_confidence_gate_preserves_strong_learned_answers():
    p, hits = apply_values([.99, .6], {"normalized": np.array([0, 0])}, confidence_floor=.05)
    assert p.tolist() == [.99, 0.]
    assert hits.tolist() == [False, True]
    with pytest.raises(ValueError):
        apply_values([.4], {}, confidence_floor=.5)
