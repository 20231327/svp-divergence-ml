"""
05A.Label Regions.py
Reads only lat/lon from the full parquet, looks up basin code
using the pre-computed 1° mask, writes a very small parquet with one
new column: region_code (int16, -1 on land/undefined).
"""
from pathlib import Path
import numpy as np
import dask.dataframe as dd

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data"
FULL  = DATA / "armor3d_full_with_svp.parquet"
MASKF = DATA / "ocean_mask_1deg.npz"
OUT   = DATA / "armor3d_region_only.parquet"

masknpz = np.load(MASKF)
MASK = masknpz["mask"]; LAT0 = masknpz["lat0"]; LON0 = masknpz["lon0"]

def add_region(df):
    lat_idx = np.floor(df["latitude"] - LAT0 + 0.5).astype("int16")
    lon_idx = np.floor((df["longitude"] - LON0 + 0.5) % 360).astype("int16")
    df["region_code"] = MASK[lat_idx, lon_idx]
    return df[["region_code"]]                # keeps memory tiny

coords = dd.read_parquet(FULL, columns=["latitude", "longitude"])
region = coords.map_partitions(add_region, meta={"region_code": "int16"})

region.to_parquet(OUT, write_index=False)
print(" region codes saved →", OUT)
