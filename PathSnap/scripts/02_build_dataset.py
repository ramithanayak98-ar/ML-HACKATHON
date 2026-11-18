"""PERSON A - Step A6: build the feature tables for every trip. Run:  python scripts/02_build_dataset.py"""
import os, sys, time, pandas as pd
sys.path.insert(0, os.getcwd())
from src.pipeline import prepare_trip
MAX = int(os.environ.get("MAX_TRIPS", "100000"))   # set MAX_TRIPS=5 for a quick test
for part in ["train", "student_test"]:
    tables, t0 = [], time.time()
    for tid in sorted(os.listdir(f"data/{part}"))[:MAX]:
        tables.append(prepare_trip(f"data/{part}", tid)[4]); print(part, tid, len(tables[-1]), "rows")
    df = pd.concat(tables, ignore_index=True)
    df.to_parquet(f"results/features_{part}.parquet", index=False)
    print(f"{part}: {len(df)} rows, positive rate {df.label.mean():.3f}, {time.time()-t0:.0f}s")
