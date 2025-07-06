"""
07C_SHAPDrilldown.py  –  master SHAP table (76 basins × depth × season)
"""
from pathlib import Path
import json, datetime as dt
import numpy as np
import pandas as pd
import regionmask

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = DATA / "shap_mean_abs_basin_depth_season.csv"

TEST_PQ   = DATA / "armor3d_test_raw.parquet"
SH_AUG_NP = DATA / "shap_values_test_aug.npz"
MASK_NPZ  = DATA / "ocean_mask_1deg.npz"
LOOKUP    = DATA / "basin_code_lookup.json"

FEATS = ["to","so","depth_val","speed","sin_dir","cos_dir","log_mld"]

# ── 0 ▸ ensure basin-number → name lookup exists ───────────────────
if not LOOKUP.exists():
    print(" building basin_code_lookup.json …")
    basins = (regionmask.defined_regions
                        .natural_earth_v5_1_2
                        .ocean_basins_50)      # ← whichever version is present

    # works for both old-style (objects) and new-style (arrays)
    try:                                     # ≤0.10.x — region objects
        code2name = {int(r.number): r.name for r in basins.regions}
    except AttributeError:                   # ≥0.11   — arrays
        code2name = {int(n): str(name)
                     for n, name in zip(basins.numbers, basins.names)}

    with LOOKUP.open("w") as f:
        json.dump(code2name, f, indent=2)
else:
    code2name = json.load(LOOKUP.open())


# ── 1 ▸ load test + SHAP -------------------------------------------
test_df  = pd.read_parquet(TEST_PQ)
shap_aug = np.load(SH_AUG_NP)["shap_values"]        # (n,7)
assert shap_aug.shape == (len(test_df), len(FEATS))

# ── 2 ▸ basin labels (vectorised, ≤20 ms) ---------------------------
m = np.load(MASK_NPZ);  MASK,LAT0,LON0 = m["mask"], m["lat0"], m["lon0"]

li = np.floor(test_df.latitude  - LAT0 + 0.5).astype("int16")
lo = np.floor((test_df.longitude - LON0 + 0.5) % 360).astype("int16")
basin = pd.Series(MASK[li, lo]).map(lambda c: code2name.get(int(c), "Unknown"))

# ── 3 ▸ depth-bin & season columns ---------------------------------
depth_bin = pd.cut(
    test_df.depth_val,
    [0,100,300,700,1200,1750,np.inf],
    labels=["0-100","100-300","300-700","700-1200","1200-1750",">1750"]
)
season = ((pd.to_datetime(test_df.time).dt.month - 1) // 3).map(
    {0:"DJF",1:"MAM",2:"JJA",3:"SON"}
)

# ── 4 ▸ tidy |SHAP| frame & aggregate ------------------------------
shap_df = (pd.DataFrame(np.abs(shap_aug), columns=FEATS)
             .assign(basin=basin.values,
                     depth_bin=depth_bin.values,
                     season=season.values)
             .query("basin != 'Unknown'"))

table = (shap_df
         .groupby(["basin","depth_bin","season"])[FEATS]
         .mean()
         .sort_index())            # Multi-index rows

table.to_csv(OUT)

print(f" SHAP table written → {OUT.relative_to(ROOT)}   rows={len(table)}")
print("  timestamp", dt.datetime.utcnow().isoformat(" ", 'seconds'))
