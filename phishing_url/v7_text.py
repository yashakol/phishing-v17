"""Full-training-side TF-IDF ablation of ContentTextModel, entirely offline.

The feature definitions and logistic classifier are inherited deliberately so
the only experimental change is the population used to learn vocabulary/IDF.
No vocabulary sampling parameter is exposed. Each fit owns fresh vectorizers.
The project's estimator convention returns a one-dimensional P(label=1).
"""

from .html_text import ContentTextModel


class FullVocabularyTextModel(ContentTextModel):
    def __init__(self, c=16.0, min_df=2, word_features=140000,
                 char_features=100000, markup_features=70000,
                 url_features=90000, char_text_chars=2000, max_iter=500,
                 threads=2, random_state=20261006,
                 transform_batch_size=2048, verbose=True):
        super().__init__(
            c=c, nb=False, include_url=True, include_char=True,
            include_markup=True, min_df=min_df, word_features=word_features,
            char_features=char_features, markup_features=markup_features,
            url_features=url_features, char_text_chars=char_text_chars,
            max_iter=max_iter, threads=threads, random_state=random_state,
            vocabulary_fit_rows=2**63 - 1,
            transform_batch_size=transform_batch_size, verbose=verbose)

    def fit(self, records, y):
        records = list(records)
        # Reset even if a caller modified an inherited attribute after init.
        self.vocabulary_fit_rows = max(2, len(records))
        super().fit(records, y)
        if self.vocabulary_fit_rows_ != len(records):
            raise AssertionError("Vocabulary/IDF must use every training row")
        self.training_rows_ = len(records)
        self.vocabulary_policy_ = "all fit rows; independent on every fold"
        return self
