"""Linear classifier over static extra HTML fields; scripts remain inert strings."""

from .html_text import ContentTextModel


class ExtraTextModel(ContentTextModel):
    def __init__(self, c=16., nb=False, threads=2, verbose=False):
        super().__init__(c=c, nb=nb, threads=threads, vocabulary_fit_rows=1000000, verbose=verbose)

    def _documents(self, records):
        return {name: [record.get(name, "") for record in records]
                for name in ("scripts", "styles", "meta", "resources")}

    def _specifications(self):
        return [(name, dict(analyzer="word", ngram_range=(1, 2),
                            token_pattern=r"(?u)[\w$:.@/-]+", max_features=150000))
                for name in ("scripts", "styles", "meta", "resources")]
