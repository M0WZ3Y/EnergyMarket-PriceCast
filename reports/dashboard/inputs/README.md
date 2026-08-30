# Pinned dashboard inputs

The generator resolves variants by EXACT filename (never basename.split("__")[0]).
Copy ONLY the specific cache CSVs the tables consume into this folder — the cache
itself in data/ablation_cache/ stays frozen and untouched. Fill the <PIN> filenames
and sha256 values in scripts/build_dashboard_data.py from the real cache once:

  git checkout -b mvp-dashboard
  python scripts/build_dashboard_data.py --print-hashes   # emit sha256 per file
  # copy the pinned CSVs here, commit them on the mvp-dashboard branch only
