"""Exercise the real command interface on eight genuine unlabeled rows."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
from pathlib import Path
import socket,sys
def blocked(*args,**kwargs):
    raise AssertionError("Network is disabled during CLI inference")
socket.create_connection=blocked
socket.getaddrinfo=blocked
import joblib,pandas as pd
from phishing_url import v10_cli
root=Path.cwd()
assert Path(v10_cli.__file__).resolve().is_relative_to(root)
out=Path(sys.argv[1])
cache=joblib.load(root/"data/content_test.joblib")
url=joblib.load(root/"data/content_url_test.joblib")
extra=joblib.load(root/"data/extra_html_test.joblib")
assert cache["labels"] is None and extra["labels"] is None
assert cache["urls"]==url["urls"]==extra["urls"]
sample={**cache,"urls":cache["urls"][:8],"records":cache["records"][:8],
        "hashes":cache["hashes"][:8],"features":cache["features"].iloc[:8].reset_index(drop=True)}
aux={**url,"urls":url["urls"][:8],"features":url["features"].iloc[:8].reset_index(drop=True)}
extra_sample={**extra,"urls":extra["urls"][:8],"records":extra["records"][:8],"features":extra["features"].iloc[:8].reset_index(drop=True)}
joblib.dump(extra_sample,out/"extra.joblib")
joblib.dump(sample,out/"cache.joblib")
joblib.dump(aux,out/"url.joblib")
pd.DataFrame({"url":sample["urls"]}).to_csv(out/"test.csv",index=False)
pd.read_csv(root/"data/sample.csv").head(8).to_csv(out/"sample.csv",index=False)
sys.argv=["v10_cli","--model",str(root/"artifacts/v10/ensemble.joblib"),
          "--content-cache",str(out/"cache.joblib"),"--extra-cache",str(out/"extra.joblib"),
          "--url-cache",str(out/"url.joblib"),"--test",str(out/"test.csv"),
          "--sample",str(out/"sample.csv"),"--output",str(out/"submission.csv")]
v10_cli.main()
pd.testing.assert_frame_equal(pd.read_csv(out/"submission.csv"),
                              pd.read_csv(root/"submissions/submission_content_v10.csv").head(8))
print("Offline CLI matches all 8 selected submission rows.")
'''


def main():
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="phishing-v10-cli-") as directory:
        result = subprocess.run([sys.executable, "-c", PROBE, directory], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        print(result.stdout, end="", flush=True)
    report = {"status": "passed", "real_test_rows": 8, "network_disabled": True,
              "predictions_match_submission": True, "actual_cli_entry_point": True}
    (ROOT / "reports/v10/cli_verification.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
