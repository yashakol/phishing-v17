"""F1 threshold selection, class-specific metrics and split integrity checks."""

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def best_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    # Prefer the optimum closest to 0.5 when a whole interval has equal F1.
    index = best[np.argmin(abs(thresholds[best] - 0.5))]
    return float(thresholds[index]), float(f1[index])


def measure(labels, probabilities, threshold=0.5):
    predictions = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"f1": float(f1_score(labels, predictions, zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "threshold": float(threshold), "rows": len(labels),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_ensemble(labels, predictions):
    candidates = []
    names = list(predictions)
    for text in [name for name in names if name != "boost"]:
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            proba = weight * predictions[text] + (1 - weight) * predictions["boost"]
            threshold, score = best_threshold(labels, proba)
            candidates.append({"text_model": text, "text_weight": weight,
                               "threshold": threshold, "development_oof_f1": score})
    # Tie-break towards balanced combinations and thresholds near 0.5.
    selected = max(candidates, key=lambda row: (row["development_oof_f1"],
                                               -abs(row["text_weight"] - 0.5), -abs(row["threshold"] - 0.5)))
    return selected, candidates
