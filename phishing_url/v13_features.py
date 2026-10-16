"""Add inert JavaScript/resource measurements to an uncertain-output specialist."""

import numpy as np
import pandas as pd


def extend_frame(base_frame, extra_features):
    if len(base_frame) != len(extra_features):
        raise ValueError("Extra specialist rows must align")
    extra = extra_features.select_dtypes(include=["number", "bool"]).reset_index(drop=True)
    result = pd.concat([base_frame.reset_index(drop=True), extra.add_prefix("static_")], axis=1)
    if result.columns.duplicated().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Extra specialist features must be unique and finite")
    return result
