import numpy as np

from phishing_url.content_memory import fit_memory, apply_memory


def record(url, text):
    return dict(url=url, title="", text=text, markup="tag_p")


def test_only_repeated_unambiguous_training_pages_override():
    training = [record("https://a.invalid", "login"), record("https://b.invalid", "login"),
                record("https://c.invalid", "generic"), record("https://d.invalid", "generic")]
    memory = fit_memory(training, ["a", "b", "c", "d"], [1, 1, 0, 1])
    p, hits = apply_memory(memory, [record("https://e.invalid", "login"),
                                    record("https://f.invalid", "generic")], ["e", "f"], [.2, .4])
    np.testing.assert_array_equal(p, [1., .4])
    np.testing.assert_array_equal(hits, [True, False])


def test_conflicting_raw_and_text_matches_keep_probability():
    records = [record("https://a.invalid", "A")]
    from phishing_url.content_memory import text_fingerprint
    memory = {("raw", "rawhash"): 0, ("text", text_fingerprint(records[0])): 1}
    p, hits = apply_memory(memory, records, ["rawhash"], [.3])
    assert p[0] == .3 and not hits[0]


def test_domain_memory_only_uses_fitted_repeated_domains():
    training = [record("https://a.example.com/a", "page A"), record("https://b.example.com/b", "page B")]
    memory = fit_memory(training, ["a", "b"], [0, 0], minimum_support=0, domain_minimum_support=2)
    testing = [record("https://c.example.com/c", "unknown"), record("https://different.org/", "unknown"),
               record("https://c.example.com/d", "confident conflict")]
    p, hits = apply_memory(memory, testing, ["c", "d", "e"], [.7, .7, .99])
    np.testing.assert_array_equal(p, [0., .7, .99])
    np.testing.assert_array_equal(hits, [True, False, False])
