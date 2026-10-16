"""Select an exploratory C32 full mixture before diagnostic holdout/test use."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v6.screen_extra import sha
from experiments.v8_ensemble.search import main as search


def main():
    output = ROOT / "reports/v8"
    source = ROOT / "experiments/v8_astra_regularization/exploratory_oof_selection.json"
    parameters = json.loads(source.read_text())["parameters"]
    candidate, p = search(ROOT / "experiments/v8_astra_regularization/development_oof.npz",
                          parameters, "exploratory_c32")
    baseline = json.loads((ROOT / "experiments/v8_ensemble/baseline_c16_selection.json").read_text())
    if candidate["selected"]["metrics"]["f1"] <= baseline["selected"]["metrics"]["f1"]:
        (output / "negative_selection.json").write_text(json.dumps({"baseline_v7": baseline["selected"],
            "c32": candidate["selected"], "new_submission_created": False,
            "external_test_labels_used": False}, indent=2) + "\n")
        print(json.dumps({"event": "v8_no_gain", "candidate": candidate["selected"], "baseline": baseline["selected"]}), flush=True)
        return
    candidate.update(exploratory_text_selection=str(source.relative_to(ROOT)),
        exploratory_text_selection_sha256=sha(source),
        public_baseline=json.loads((output / "baseline_public_feedback.json").read_text()),
        new_astra_calls=1,
        exploratory_note="Original simple-mixture gate failed. Chosen C32 standalone reduced inner errors by two; one explicitly frozen exploratory OOF branch checked full mixture. OOF used for selection; no external test labels.")
    (output / "selection.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(json.dumps({"event": "v8_frozen", "selected": candidate["selected"], "baseline_v7": candidate["baseline_v7"]}), flush=True)


if __name__ == "__main__":
    main()
