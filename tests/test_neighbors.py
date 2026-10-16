import numpy as np

from phishing_url.neighbors import NeighborModel, adjust_neighbors


def test_neighbor_labels_match_the_retrieved_training_pages():
    records = [{"url": "https://one.invalid", "title": "", "text": "science article safe"},
               {"url": "https://two.invalid", "title": "", "text": "science article safe"},
               {"url": "https://three.invalid", "title": "", "text": "password account verify"},
               {"url": "https://four.invalid", "title": "", "text": "password account verify"}]
    model = NeighborModel(view="content", max_neighbors=2).fit(records, [0, 0, 1, 1])
    similarities, labels = model.lookup([{"url": "https://unseen.invalid", "title": "", "text": "password account verify"}])
    np.testing.assert_allclose(similarities, 1, atol=1e-6)
    np.testing.assert_array_equal(labels, [[1, 1]])


def test_no_matches_and_hard_probabilities_are_preserved():
    base = np.array([.3, 0., 1.])
    similarities = np.array([[0., 0.], [1., 1.], [1., 1.]])
    labels = np.array([[-1, -1], [1, 1], [0, 0]])
    result = adjust_neighbors(base, similarities, labels, {"minimum_similarity": .8, "weight": .6})
    np.testing.assert_array_equal(result, base)
