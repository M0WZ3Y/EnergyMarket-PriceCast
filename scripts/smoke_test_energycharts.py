"""Week-1 smoke test for the Energy-Charts live API (keyless).

    python scripts/smoke_test_energycharts.py [START] [END]

Defaults to the first week of last month. Confirms:
  - the /price endpoint responds for the configured bidding zone
  - the JSON parses into the standardized schema
  - resolution before/after hourly resampling
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import EnergyChartsLoader, load_config


def main() -> None:
    first_of_this_month = date.today().replace(day=1)
    last_month_start = (first_of_this_month - timedelta(days=1)).replace(day=1)
    start = sys.argv[1] if len(sys.argv) > 1 else str(last_month_start)
    end = sys.argv[2] if len(sys.argv) > 2 else str(last_month_start + timedelta(days=7))

    cfg = load_config()
    loader = EnergyChartsLoader(cfg)
    print(f"Zone: {cfg['live']['bzn']}   Range: {start} -> {end}")

    df = loader.fetch_prices(start=start, end=end)

    print(f"Rows (hourly):    {len(df)}")
    print(f"Date range:       {df.index.min()}  ->  {df.index.max()}")
    print(f"Price stats:      mean={df['price'].mean():.2f}  "
          f"min={df['price'].min():.2f}  max={df['price'].max():.2f} EUR/MWh")
    print(f"\nHead:\n{df.head()}")
    print(f"\nAttribution required: {loader.attribution}")
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
