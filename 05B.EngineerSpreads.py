"""
05B.EngineerSpreads.py

"""
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FULL  = DATA / "armor3d_full_with_svp.parquet"
MASKF = DATA / "ocean_mask_1deg.npz"
OUT   = DATA / "armor3d_spreads.parquet"

FORMULAS     = ["svp_mackenzie", "svp_coppens", "svp_unesco",
                "svp_del_grosso", "svp_npl"]

DEPTH_BINS   = [0, 100, 300, 700, 1200, 1750]
DEPTH_LABELS = ["0–100", "100–300", "300–700",
                "700–1200", "1200–1750"]
depth_dtype  = pd.api.types.CategoricalDtype(
                  categories=DEPTH_LABELS, ordered=True
               )

# ---------- load 1° mask (tiny) ------------------------------------
npz   = np.load(MASKF)
MASK  = npz["mask"];  LAT0 = npz["lat0"];  LON0 = npz["lon0"]

# ---------- read parquet in small partitions -----------------------
cols = ["latitude","longitude","depth_val","time",
        "svp_spread_max","svp_spread_std"] + FORMULAS

ddf = dd.read_parquet(
        FULL, columns=cols, engine="pyarrow",
        split_row_groups=True, gather_statistics=False
)

# ---------- per-partition engineering ------------------------------
def engineer(df):
    lat_idx = np.floor(df["latitude"].to_numpy()  - LAT0 + 0.5).astype("int16")
    lon_idx = np.floor((df["longitude"].to_numpy() - LON0 + 0.5) % 360).astype("int16")
    region_code = MASK[lat_idx, lon_idx]

    depth_val = df["depth_val"].astype("float32")
    depth_bin = pd.cut(depth_val, DEPTH_BINS, labels=DEPTH_LABELS).astype(depth_dtype)
    season    = ((df["time"].dt.month - 1) // 3).astype("int8") % 4

    out = pd.DataFrame({
        "depth_val":        depth_val.values,
        "svp_spread_max":   df["svp_spread_max"].astype("float32").values,
        "svp_spread_std":   df["svp_spread_std"].astype("float32").values,
        "region_code":      region_code,
        "depth_bin":        depth_bin.values,
        "season":           season.values,
    })

    for col in FORMULAS:
        out[col] = df[col].astype("float32").values

    mean_svp = out[FORMULAS].mean(axis=1).to_numpy()
    for col in FORMULAS:
        out[f"dev_{col}"] = np.abs(out[col].to_numpy() - mean_svp).astype("float32")

    return out

# ---------- exact-order meta ---------------------------------------
out_cols = [
    "depth_val","svp_spread_max","svp_spread_std",
    "region_code","depth_bin","season",
    "svp_mackenzie","svp_coppens","svp_unesco","svp_del_grosso","svp_npl",
    "dev_svp_mackenzie","dev_svp_coppens","dev_svp_unesco",
    "dev_svp_del_grosso","dev_svp_npl",
]

meta = pd.DataFrame({c: pd.Series(dtype="float32") for c in out_cols})
meta["region_code"] = pd.Series(dtype="int16")
meta["depth_bin"]   = pd.Series(dtype=depth_dtype)
meta["season"]      = pd.Series(dtype="int8")

# ---------- map, persist, write ------------------------------------
ddf = ddf.map_partitions(engineer, meta=meta).persist()
ddf.to_parquet(OUT, write_index=False)
print(" engineered spreads saved →", OUT)
