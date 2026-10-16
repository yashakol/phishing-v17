from phishing_url.extra_text import ExtraTextModel


def test_extra_text_vocabulary_stays_training_only():
    records = [{"meta": "normal science article safe"}, {"meta": "normal science article safe"},
               {"scripts": "password account verify login"}, {"scripts": "password account verify login"}]
    model = ExtraTextModel(c=16).fit(records, [0, 0, 1, 1])
    assert model.vocabulary_fit_rows_ == 4
    vocabulary = {name: dict(vectorizer.vocabulary_) for name, vectorizer in model.vectorizers_.items()}
    result = model.predict_proba([{"meta": "onlynewtestword", "scripts": "anothernewtestword"}])
    assert result.shape == (1,)
    assert all(vectorizer.vocabulary_ == vocabulary[name] for name, vectorizer in model.vectorizers_.items())


def test_extra_text_handles_empty_fields_and_returns_class_one():
    records = [{"meta": "normal science article safe"}, {"meta": "normal science article safe"},
               {"scripts": "password account verify login"}, {"scripts": "password account verify login"}]
    model = ExtraTextModel(c=16).fit(records, [0, 0, 1, 1])
    p = model.predict_proba(records)
    assert max(p[:2]) < .5 < min(p[2:])
