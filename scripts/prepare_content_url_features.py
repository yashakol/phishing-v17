"""Cache URL features in the uncorrupted archive's training order."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib

from phishing_url.data import load_train, load_test
from phishing_url.features import make_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    train, labels, _ = load_train(args.archive / "train.csv")
    test = load_test(args.archive / "test.csv")
    for name, urls in (("train", train), ("test", test)):
        print("Extracting URL features:", name, len(urls), flush=True)
        frame = make_features(urls)
        joblib.dump({"urls": urls, "features": frame}, ROOT / f"data/content_url_{name}.joblib", compress=3)
        print("Saved", name, frame.shape, flush=True)


if __name__ == "__main__":
    main()
