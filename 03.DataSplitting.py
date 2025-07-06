"""
03_DataSplitting.py
Reads  data/armor3d_strat_sample_with_svp.parquet  (≈48 300 rows × all cols)
Builds spatial+season stratification keys
Produces 80 / 10 / 10 train-val-test Parquets (retaining all SVP & spread cols)
"""

# ── 1 ▸ imports & load sample ───────────────────────────────────────
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent       # project root
DATA = ROOT / "data"
IN_PQ = DATA / "armor3d_strat_sample_with_svp.parquet"

assert IN_PQ.exists(), f"sample parquet not found: {IN_PQ}"
print(" Reading", IN_PQ.relative_to(ROOT))
df = pd.read_parquet(IN_PQ)
print("   rows:", len(df), "cols:", len(df.columns))

# ── 2 ▸ build stratification key  (spatial k-means  +  season) ──────
print(" Building stratification key …")

# 2.1  spatial clusters on (lat, lon)
coords = df[["latitude", "longitude"]].to_numpy(dtype=np.float32)
k = 40
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(coords)
df["sp_cluster"] = kmeans.labels_.astype(np.int16)

# 2.2  season 0-3  (DJF=0 … SON=3)
df["season_q"] = ((pd.to_datetime(df["time"]).dt.month - 1) // 3).astype(np.int8)

# 2.3  final key  cluster*10 + season  (10 ≥ 4 keeps keys unique)
df["strat_key"] = df["sp_cluster"] * 10 + df["season_q"]

uniq, counts = np.unique(df["strat_key"], return_counts=True)
print("   unique keys:", uniq.size, " | min / max count:",
      counts.min(), counts.max())

# ── 3 ▸ 80 / 10 / 10 split  (per-key proportional) ──────────────────
print(" Splitting …")

train_idx, val_idx, test_idx = [], [], []
rng = np.random.default_rng(42)

for key, sub in df.groupby("strat_key").indices.items():
    idx = np.fromiter(sub, dtype=np.int64)
    rng.shuffle(idx)

    n = len(idx)
    if n <= 2:                         # tiny strata → keep all in train
        train_idx.extend(idx)
        continue

    n_val   = max(1, int(round(n * 0.10)))
    n_test  = max(1, int(round(n * 0.10)))
    n_train = n - n_val - n_test

    train_idx.extend(idx[:n_train])
    val_idx.extend(idx[n_train:n_train + n_val])
    test_idx.extend(idx[n_train + n_val:])

print(f"   rows  train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

# ▸ helper to write a parquet ----------------------------------------
def _write(name: str, indices: list[int]) -> Path:
    out = DATA / f"armor3d_{name}.parquet"
    df.loc[indices].to_parquet(out, index=False)
    print(f"    wrote {name:<5s}  →  {out.relative_to(ROOT)}")
    return out

TRAIN_PQ = _write("train", train_idx)
VAL_PQ   = _write("val",   val_idx)
TEST_PQ  = _write("test",  test_idx)

# ── 4 ▸ quick sanity prints ─────────────────────────────────────────
print("\n Split distribution (rows per key) – first 8 keys:")
split_counts = (
    pd.concat([
        df.loc[train_idx, ["strat_key"]].assign(split="train"),
        df.loc[val_idx,   ["strat_key"]].assign(split="val"),
        df.loc[test_idx,  ["strat_key"]].assign(split="test"),
    ])
    .groupby(["strat_key", "split"]).size().unstack(fill_value=0)
)
print(split_counts.head(8))

print("\n Done.")
