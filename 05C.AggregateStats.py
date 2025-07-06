"""
05C. AggregateStats.py
reads armor3d_spreads.parquet
aggregates by (region_code, depth_bin, season)
    – group-mean for every numeric column
    – 95th-percentile for svp_spread_max
writes data/armor3d_stats.parquet
"""

from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import numpy as np

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data"
SRC   = DATA / "armor3d_spreads.parquet"
DEST  = DATA / "armor3d_stats.parquet"

print(" loading spreads")
ddf = dd.read_parquet(SRC, engine="pyarrow", gather_statistics=False)


# ------------------------------------------------------------------
# 1) cast through the known ordered categorical to normalise values
DEPTH_BINS = ["0–100", "100–300", "300–700", "700–1200", "1200–1750"]
DEPTH_DTYPE = pd.CategoricalDtype(categories=DEPTH_BINS, ordered=True)
ddf["depth_bin"] = ddf["depth_bin"].astype(DEPTH_DTYPE)

# 2) immediately drop categorical metadata → ordinary string
ddf["depth_bin"] = ddf["depth_bin"].astype(str)

# ------------------------------------------------------------------
# group keys & numeric columns
# ------------------------------------------------------------------
GROUP_KEYS = ["region_code", "depth_bin", "season"]

NUMERIC_COLS = [
    "svp_spread_max", "svp_spread_std",
    "svp_mackenzie", "svp_coppens", "svp_unesco",
    "svp_del_grosso", "svp_npl",
    "dev_svp_mackenzie", "dev_svp_coppens", "dev_svp_unesco",
    "dev_svp_del_grosso", "dev_svp_npl",
]

# ------------------------------------------------------------------
# 1) group-mean for every numeric column
# ------------------------------------------------------------------
print(" computing group means")
mean_dd = (
    ddf[GROUP_KEYS + NUMERIC_COLS]
    .groupby(GROUP_KEYS)
    .mean()
)

print("    materialising")
mean_df = mean_dd.compute()
mean_df.columns = [f"{c}_mean" for c in mean_df.columns]

# ------------------------------------------------------------------
# 2) 95th-percentile for svp_spread_max  (approx.; TDigest)
# ------------------------------------------------------------------
print(" computing p95 (svp_spread_max)")
def q95(partition):
    return partition.quantile(0.95, interpolation="nearest")

p95_dd = (
    ddf[GROUP_KEYS + ["svp_spread_max"]]
    .groupby(GROUP_KEYS)["svp_spread_max"]
    .apply(q95, meta=("svp_spread_max", "float32"))
)

print("    materialising")
p95_df = p95_dd.compute().to_frame(name="svp_spread_max_p95")

# ------------------------------------------------------------------
# merge & save
# ------------------------------------------------------------------
print(" merging results")
stats = (
    mean_df
    .reset_index()
    .merge(p95_df.reset_index(), on=GROUP_KEYS)
    .set_index(GROUP_KEYS)
    .sort_index()
)

print(" writing parquet")
stats.to_parquet(DEST, engine="pyarrow", index=True)
print(f" aggregated stats written → {DEST}")
