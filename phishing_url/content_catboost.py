"""Native CatBoost text features combined with static HTML/URL features."""

import re

from catboost import CatBoostClassifier

from .features import categorical_columns, text_tokens

TEXT_COLUMNS = ["text_title", "text_visible", "text_markup", "text_url"]
WORD = re.compile(r"[\w:-]+", re.UNICODE)


def native_frame(frame, records):
    out = frame.copy()
    for column, key, limit in (("text_title", "title", 600), ("text_visible", "text", 12000),
                               ("text_markup", "markup", 16000)):
        out[column] = [" ".join(WORD.findall(record[key][:limit].lower())) or "empty_document" for record in records]
    out["text_url"] = [" ".join(text_tokens(record["url"])) for record in records]
    return out


class NativeContentBoost:
    def __init__(self, iterations=1800, depth=7, learning_rate=.06, l2_leaf_reg=5,
                 bow_tokens=0, threads=2):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.bow_tokens = bow_tokens
        self.threads = threads

    def fit(self, frame, records, labels, evaluation=None):
        transformed = native_frame(frame, records)
        calcers = ["NaiveBayes", "BM25"]
        if self.bow_tokens:
            calcers.append(f"BoW:top_tokens_count={self.bow_tokens}")
        self.model = CatBoostClassifier(iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            random_seed=2026, thread_count=self.threads, loss_function="Logloss",
            verbose=False, allow_writing_files=False,
            tokenizers=[{"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " ", "lowercasing": "true"}],
            dictionaries=[{"dictionary_id": "Word", "gram_order": "1", "max_dictionary_size": "50000", "occurrence_lower_bound": "2"},
                          {"dictionary_id": "Bigram", "gram_order": "2", "max_dictionary_size": "50000", "occurrence_lower_bound": "2"}],
            feature_calcers=calcers)
        options = {"cat_features": categorical_columns(frame), "text_features": TEXT_COLUMNS}
        if evaluation is not None:
            va_frame, va_records, va_y = evaluation
            options.update(eval_set=(native_frame(va_frame, va_records), va_y), early_stopping_rounds=120, use_best_model=True)
        self.model.fit(transformed, labels, **options)
        self.iterations = int(self.model.tree_count_)
        return self

    def predict_proba(self, frame, records):
        return self.model.predict_proba(native_frame(frame, records))[:, 1]

    def configuration(self):
        return {name: getattr(self, name) for name in ("iterations", "depth", "learning_rate", "l2_leaf_reg", "bow_tokens", "threads")}
