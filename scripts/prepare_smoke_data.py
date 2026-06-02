#!/usr/bin/env python3
"""Create a small smoke-test dataset from the full training data."""
import pandas as pd
from pathlib import Path


def balanced_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Return an exact-size deterministic sample balanced across data sources."""
    sample_size = min(sample_size, len(df))
    if "data_source" not in df.columns:
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    groups = [group for _, group in df.groupby("data_source", sort=True)]
    quota, remainder = divmod(sample_size, len(groups))
    parts = []
    for index, group in enumerate(groups):
        count = min(len(group), quota + (1 if index < remainder else 0))
        parts.append(group.sample(n=count, random_state=42))

    sample = pd.concat(parts)
    if len(sample) < sample_size:
        remaining = df.drop(index=sample.index)
        sample = pd.concat([
            sample,
            remaining.sample(n=sample_size - len(sample), random_state=43),
        ])
    return sample.sample(frac=1, random_state=42).reset_index(drop=True)


data_dir = Path(__file__).resolve().parent.parent / "data"
src = data_dir / "train.parquet"
dst_dir = data_dir / "smoke"
dst_dir.mkdir(exist_ok=True)

df = pd.read_parquet(src)
print(f"Full train set: {len(df)} rows")

# Sample exactly 200 rows, balanced by data_source if possible
sample = balanced_sample(df, 200)

# Also create a tiny 20-row set for ultra-fast debugging
tiny = df.sample(n=min(20, len(df)), random_state=42)

sample.to_parquet(dst_dir / "train_200.parquet", index=False)
tiny.to_parquet(dst_dir / "train_20.parquet", index=False)

# Copy dev as-is for validation
dev = data_dir / "dev.parquet"
if dev.exists():
    dev_df = pd.read_parquet(dev)
    dev_sample = dev_df.sample(n=min(20, len(dev_df)), random_state=42)
    dev_sample.to_parquet(dst_dir / "dev_20.parquet", index=False)
    print(f"Smoke dev: {len(dev_sample)} rows")

print(f"Smoke train (200): {len(sample)} rows")
print(f"Smoke train (20):  {len(tiny)} rows")
print(f"Saved to {dst_dir}")
