"""An own-trained specialist for the uncertain outputs of a frozen ensemble."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit


def eligibility(baseline):
    p = np.asarray(baseline, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Baseline probabilities must be finite and binary-class probabilities")
    return (p >= .05) & (p <= .95)


def expert_frame(meta, numeric, baseline):
    if len(meta) != len(numeric) or len(meta) != len(baseline):
        raise ValueError("Expert inputs must have aligned rows")
    eligibility(baseline)
    numeric = numeric.select_dtypes(include=["number", "bool"])
    result = pd.concat([meta.reset_index(drop=True), numeric.reset_index(drop=True).add_prefix("numeric_")], axis=1)
    result["baseline_probability"] = np.asarray(baseline)
    result["baseline_logit"] = logit(np.clip(baseline, 1e-6, 1-1e-6))
    if result.columns.duplicated().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Expert schema and values must be unique and finite")
    return result


def make_expert():
    return CatBoostClassifier(depth=5, iterations=800, learning_rate=.04, l2_leaf_reg=30.,
        loss_function="Logloss", random_seed=20261010, thread_count=2, verbose=False, allow_writing_files=False)


def blend_expert(baseline, expert, weight):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid expert mixture weight")
    p, q = np.asarray(baseline, dtype=float), np.asarray(expert, dtype=float)
    mask = eligibility(p)
    if p.shape != q.shape or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("Expert probabilities must align")
    result = p.copy()
    result[mask] = (1-weight)*p[mask]+weight*q[mask]
    return result
