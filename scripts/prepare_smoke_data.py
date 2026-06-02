#!/usr/bin/env python3
"""Create a small smoke-test dataset from the full training data."""
import pandas as pd
import sys
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "data"
src = data_dir / "train.parquet"
dst_dir = data_dir / "smoke"
dst_dir.mkdir(exist_ok=True)

df = pd.read_parquet(src)
print(f"Full train set: {len(df)} rows")

# Sample 200 rows, stratified by data_source if possible
sample = df.groupby("data_source", group_keys=False).apply(
    lambda x: x.sample(n=min(67, len(x)), random_state=42)
).reset_index(drop=True)

# Also create a tiny 20-row set for ultra-fast debugging
tiny = df.sample(n=20, random_state=42)

sample.to_parquet(dst_dir / "train_200.parquet", index=False)
tiny.to_parquet(dst_dir / "train_20.parquet", index=False)

# Copy dev as-is for validation
dev = data_dir / "dev.parquet"
if dev.exists():
    dev_sample = pd.read_parquet(dev).sample(n=20, random_state=42)
    dev_sample.to_parquet(dst_dir / "dev_20.parquet", index=False)
    print(f"Smoke dev: {len(dev_sample)} rows")

print(f"Smoke train (200): {len(sample)} rows")
print(f"Smoke train (20):  {len(tiny)} rows")
print(f"Saved to {dst_dir}")
