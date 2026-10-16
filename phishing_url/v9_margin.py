"""Calibrated linear SVM on a guarded, training-fitted HTML representation."""

import copy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


class CalibratedMarginText(BaseEstimator):
    def __init__(self, c=.5, calibration_folds=3, max_iter=3000, tol=.001,
                 random_state=20261008, threads=2):
        self.c = c
        self.calibration_folds = calibration_folds
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.threads = threads

    def fit(self, records, labels, representation=None):
        if representation is None:
            raise ValueError("A guarded training-fitted representation must be supplied")
        return self.fit_representation(representation, representation._matrix(records), labels)

    def fit_representation(self, original, matrix, labels):
        y = np.asarray(labels)
        if matrix.shape != (len(y), original.n_features_) or original.vocabulary_fit_rows_ != len(y):
            raise ValueError("Frozen vocabulary and coefficient training rows must match")
        if not np.array_equal(np.unique(y), [0, 1]) or self.threads not in (1, 2):
            raise ValueError("Binary labels and bounded threads required")
        if original.nb or original.ratio_ is not None:
            raise ValueError("Unsupervised TF-IDF representation required")
        self.transformer_ = copy.deepcopy(original)
        # The old LR is not used: retain the partition's vectorizers and IDF.
        del self.transformer_.classifier_
        svm = LinearSVC(C=self.c, dual=True, max_iter=self.max_iter, tol=self.tol,
                        random_state=self.random_state)
        splitter = StratifiedKFold(self.calibration_folds, shuffle=True, random_state=self.random_state)
        self.classifier_ = CalibratedClassifierCV(svm, method="sigmoid", cv=splitter, ensemble=False, n_jobs=1)
        with threadpool_limits(limits=self.threads):
            self.classifier_.fit(matrix, y)
        self.classes_ = self.classifier_.classes_
        self.training_rows_ = len(y)
        self.n_features_ = matrix.shape[1]
        self.classifier_fits_ = self.calibration_folds+1
        self.calibration_scope_ = "Three internal training folds; vocabulary fitted on complete outer training side only"
        if not np.isfinite(self.classifier_.calibrated_classifiers_[0].estimator.coef_).all():
            raise ValueError("Nonfinite margin coefficients")
        return self

    def predict_proba(self, records):
        check_is_fitted(self, ["classifier_", "transformer_"])
        if len(records) == 0:
            return np.empty(0, dtype=float)
        with threadpool_limits(limits=self.threads):
            return self.classifier_.predict_proba(self.transformer_._matrix(records))[:, 1]
