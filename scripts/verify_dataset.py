"""Week-1 verification: download the benchmark dataset and print a
data-quality report. Run from the repo root:

    python scripts/verify_dataset.py

Success criteria (log the outcome in logs/decisions.md):
  - download completes and caches to data/raw/
  - ~6 years of hourly rows, no gaps
  - negative prices present (expected for DE)
If the epftoolbox server is unreachable, fall back to pinning the
mirrored benchmark CSVs into data/raw/ (see README).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.loader import BenchmarkLoader, load_config


def report(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"Shape:            {df.shape}")
    print(f"Date range:       {df.index.min()}  ->  {df.index.max()}")
    print(f"Columns:          {list(df.columns)}")

    # Gap check: hourly index should be perfectly regular
    expected = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    missing = expected.difference(df.index)
    print(f"Missing hours:    {len(missing)}")
    if len(missing) > 0:
        print(f"  first few:      {list(missing[:5])}")

    print(f"NaNs per column:  {df.isna().sum().to_dict()}")

    p = df["price"]
    print(f"Price stats:      mean={p.mean():.2f}  std={p.std():.2f}  "
          f"min={p.min():.2f}  max={p.max():.2f}")
    neg = (p < 0).sum()
    print(f"Negative prices:  {neg} hours ({100 * neg / len(p):.2f}%)")


def main() -> None:
    cfg = load_config()
    print(f"Dataset: {cfg['benchmark']['dataset']}  "
          f"(test years: {cfg['benchmark']['years_test']})")
    print("Downloading / reading from cache ...")

    train, test = BenchmarkLoader(cfg).load()
    report("TRAIN", train)
    report("TEST", test)

    print("\nVerification complete. Record the outcome in logs/decisions.md.")


if __name__ == "__main__":
    main()
