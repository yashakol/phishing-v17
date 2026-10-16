"""Check split identity, saved metrics and fresh-process model prediction parity."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import json
import csv
import joblib
import numpy as np
import sklearn
import scipy
from sklearn.model_selection import train_test_split, StratifiedKFold
from threadpoolctl import threadpool_limits
from phishing_url.data import load_train, load_test, sha256
from phishing_url.evaluation import measure
import url_model


def main():
    urls, labels, _ = load_train(ROOT / "data/train.csv")
    dev = np.load(ROOT / "reports/split_indices.npz")["development"]
    dev_urls = np.asarray(urls, dtype=object)[dev]
    y = labels[dev]
    del urls, labels
    positions = {int(i): j for j, i in enumerate(dev)}
    train, valid = train_test_split(dev, test_size=.2, stratify=y, random_state=20261006)
    audit = np.load(OUT / "audit_predictions.npz")
    np.testing.assert_array_equal(valid, audit["indices"])
    np.testing.assert_array_equal(train, audit["train_indices"])
    assert not set(train).intersection(valid)
    for key in ("train_indices", "tune_indices", "indices"):
        assert set(audit[key]).issubset(set(dev))
    selection = json.loads((OUT / "selection.json").read_text())
    results = json.loads((OUT / "results.json").read_text())
    model = joblib.load(OUT / "internal_selected.joblib")
    url_slice = [dev_urls[positions[int(i)]] for i in audit["indices"][:128]]
    np.testing.assert_allclose(model.predict_proba(url_slice), audit["selected"][:128], rtol=0, atol=1e-12)
    metric = measure(audit["y"], audit["selected"], selection["selected"]["threshold"])
    assert metric == results["audit"]["selected"]["all"]
    oof = np.load(OUT / "development_oof.npz")
    np.testing.assert_array_equal(oof["indices"], dev)
    np.testing.assert_array_equal(oof["y"], y)
    for f, (_, v) in enumerate(StratifiedKFold(3, shuffle=True, random_state=2026).split(dev, y)):
        assert (oof["fold"][v] == f).all()
    assert np.isfinite(oof["probabilities"]).all()
    assert ((oof["probabilities"] >= 0) & (oof["probabilities"] <= 1)).all()
    oof_results = json.loads((OUT / "oof_results.json").read_text())
    assert measure(y, oof["probabilities"], selection["selected"]["threshold"]) == oof_results["oof"]
    model = joblib.load(OUT / "development_model.joblib")
    test = load_test(ROOT / "data/test.csv")
    stored = np.load(OUT / "test_predictions.npz")
    np.testing.assert_allclose(model.predict_proba(test[:128]), stored["probabilities"][:128], rtol=0, atol=1e-12)
    with (ROOT / "data/sample.csv").open() as f:
        sample = list(csv.DictReader(f))
    with (OUT / "test_submission.csv").open() as f:
        submitted = list(csv.DictReader(f))
    assert len(test) == len(stored["probabilities"]) == len(submitted)
    assert [r["Id"] for r in sample] == [r["Id"] for r in submitted]
    np.testing.assert_array_equal([int(r["Predicted"]) for r in submitted],
                                  stored["probabilities"] >= stored["threshold"])
    result = {"verified": True, "audit_rows": len(audit["y"]), "oof_rows": len(y), "test_rows": len(test),
              "model_reload_prediction_max_atol": 1e-12,
              "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__},
              "code_sha256": {p.name: sha256(p) for p in OUT.glob("*.py")}}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
