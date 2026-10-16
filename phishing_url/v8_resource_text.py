"""Learn URL fragments inside inert HTML resources, metadata and script/style.

Resource strings remain plain text; this module never visits a URL or executes
JavaScript. All TF-IDF statistics are fitted on the provided training rows.
"""

from .html_text import ContentTextModel


class ResourceCharacterTextModel(ContentTextModel):
    def __init__(self, c=4., min_df=2, threads=2, random_state=20261008,
                 transform_batch_size=2048, verbose=True):
        super().__init__(c=c, min_df=min_df, threads=threads, random_state=random_state,
                         vocabulary_fit_rows=1000000, transform_batch_size=transform_batch_size,
                         verbose=verbose)

    def _documents(self, records):
        limits = {"resources": 16000, "meta": 6000, "scripts": 9000, "styles": 9000}
        fields = {key: [] for key in limits}
        for row in records:
            for key, limit in limits.items():
                value = row.get(key, "") or ""
                if not isinstance(value, str):
                    raise TypeError(f"record[{key!r}] must be a string")
                fields[key].append(value[:limit])
        return fields

    def _specifications(self):
        return [
            ("resources", dict(analyzer="char", ngram_range=(3, 5), max_features=200000)),
            ("meta", dict(analyzer="char_wb", ngram_range=(3, 5), max_features=100000)),
            ("scripts", dict(analyzer="word", ngram_range=(1, 2),
                             token_pattern=r"(?u)[\w$:.@/-]+", max_features=120000)),
            ("styles", dict(analyzer="word", ngram_range=(1, 2),
                            token_pattern=r"(?u)[\w$:.@/-]+", max_features=80000)),
        ]
