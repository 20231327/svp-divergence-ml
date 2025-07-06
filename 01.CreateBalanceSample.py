"""
01_CreateBalanceSample.py
Create a 54 000-row, perfectly balanced 4-way stratified sample
(depth_bin × lat_bin × lon_band × season) from the 19 ARMOR-3D subsets.
Retains geo–temporal coordinates for later regional analysis.
"""

# ── 1 ▸ imports & paths ─────────────────────────────────────────────
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

SUBSET_DIR        = Path("../data/subsets")
OUT_PQ            = Path("../data/armor3d_strat_sample.parquet")
ROWS_PER_BUCKET   = 250            # 216 × 250 = 54 000

nc_files = sorted(SUBSET_DIR.glob("*.nc"))
print(f"🗂  found {len(nc_files)} subset files")

# ── 2 ▸ helper fns (NumPy in → NumPy out)  ──────────────────────────
def depth_bin(depth):
    out = np.full_like(depth, 2, dtype="uint8")
    out[depth <= 800] = 1
    out[depth <= 200] = 0
    return out

def lat_bin(lat):
    a   = np.abs(lat)
    out = np.full_like(a, 2, dtype="uint8")
    out[a <= 45] = 1
    out[a <= 23] = 0
    return out

def lon_band(lon):                     # 6 equal 60° sectors
    return ((lon + 180) // 60).astype("uint8").clip(0, 5)

def season_bin(month):                 # 0:DJF … 3:SON
    return (((month - 1) // 3) % 4).astype("uint8")

# ── 3 ▸ pass-1  – row counts per stratum  ───────────────────────────
COUNTS = np.zeros((3, 3, 6, 4), dtype=np.int64)

for fn in tqdm(nc_files, desc="pass-1 count"):
    with xr.open_dataset(fn) as ds:
        d_m, lat, lon = ds.depth.values, ds.latitude.values, ds.longitude.values
        mons          = ds.time.dt.month.values

        mon_g, d_g, lat_g, lon_g = np.meshgrid(
            mons, d_m, lat, lon, indexing="ij", sparse=False
        )
        idx = (
            depth_bin(d_g)
          + lat_bin(lat_g)    * 3
          + lon_band(lon_g)   * 9
          + season_bin(mon_g) * 54
        ).ravel()

        COUNTS.ravel()[:216] += np.bincount(idx, minlength=216)

print(f" pass-1 done – min/median/max rows per bucket:"
      f" {COUNTS.min():,} / {int(np.median(COUNTS)):,} / {COUNTS.max():,}")

# ── 4 ▸ quota per bucket  -------------------------------------------
QUOTA = np.minimum(COUNTS, ROWS_PER_BUCKET).astype(np.int64)

# ── 5 ▸ pass-2  – sample rows  --------------------------------------
rng, frames = np.random.default_rng(42), []

for fn in tqdm(nc_files, desc="pass-2 sample"):
    with xr.open_dataset(fn) as ds:
        d_m, lat, lon = ds.depth.values, ds.latitude.values, ds.longitude.values
        mons          = ds.time.dt.month.values

        mon_g, d_g, lat_g, lon_g = np.meshgrid(
            mons, d_m, lat, lon, indexing="ij", sparse=False
        )
        strata = (
            depth_bin(d_g)
          + lat_bin(lat_g)    * 3
          + lon_band(lon_g)   * 9
          + season_bin(mon_g) * 54
        ).astype("uint16")

        flat_idx   = np.arange(strata.size)
        strata_flat = strata.ravel()

        take_mask_flat = np.zeros_like(strata_flat, bool)
        for s in np.unique(strata_flat):
            need = int(QUOTA.ravel()[s])
            if need == 0:
                continue
            pool   = flat_idx[strata_flat == s]
            chosen = rng.choice(pool, size=min(need, len(pool)),
                                replace=False)
            take_mask_flat[chosen] = True
            QUOTA.ravel()[s]     -= len(chosen)

        if not take_mask_flat.any():
            continue

        mask_da = xr.DataArray(
            take_mask_flat.reshape(mon_g.shape),
            coords=dict(time=ds.time, depth=ds.depth,
                        latitude=ds.latitude, longitude=ds.longitude),
            dims=("time", "depth", "latitude", "longitude"),
        )

        sub  = ds[["to", "so", "ugo", "vgo", "mlotst"]].where(mask_da)
        flat = sub.stack(z=("time", "depth", "latitude", "longitude"))
        flat = flat.dropna(dim="z", how="any")
        df   = flat.to_dataframe().reset_index(drop=True)
        df.rename(columns={"depth": "depth_val"}, inplace=True)

        # stratification keys (kept purely for diagnostics)
        df["depth_bin"] = depth_bin(df["depth_val"].values)
        df["lat_bin"]   = lat_bin(df["latitude"].values)
        df["lon_band"]  = lon_band(df["longitude"].values)
        df["season"]    = season_bin(df["time"].dt.month.values)

        frames.append(df)

sample = (pd.concat(frames, ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True))

print(f"  final sample rows = {len(sample):,}")

OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
sample.to_parquet(OUT_PQ, index=False)
sample.to_csv(OUT_PQ.with_suffix(".csv"), index=False)
print("  saved →", OUT_PQ)
