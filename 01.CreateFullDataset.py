"""
01.CreateFullDataset.py
--------------------------------
Streams all 19 ARMOR-3D subset files into one “full” Parquet
Keeps (time, depth, latitude, longitude) as columns
Outputs: data/armor3d_full.parquet (+ .csv if you really need it)
"""

from pathlib import Path
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

SUBSET_DIR   = Path("../data/subsets")
OUT_FULL_PQ  = Path("../data/armor3d_full.parquet")
OUT_FULL_CSV = OUT_FULL_PQ.with_suffix(".csv")  # optional

nc_files = sorted(SUBSET_DIR.glob("*.nc"))
print(f"  found {len(nc_files)} subset files to merge")

# we'll build a ParquetWriter on the fly
writer = None

for fn in tqdm(nc_files, desc="streaming to Parquet"):
    with xr.open_dataset(fn) as ds:
        ds_sel = ds[["to","so","ugo","vgo","mlotst"]]
        # convert to DataFrame with coords as columns
        df = ds_sel.to_dataframe().reset_index()
        df = df.rename(columns={"depth":"depth_val"})
        # first file: open writer with schema
        if writer is None:
            table = pa.Table.from_pandas(df, preserve_index=False)
            writer = pq.ParquetWriter(str(OUT_FULL_PQ), table.schema)
        # append this chunk
        table = pa.Table.from_pandas(df, preserve_index=False)
        writer.write_table(table)

# close Parquet
if writer is not None:
    writer.close()
    print(f" full merged dataset saved → {OUT_FULL_PQ}")
    # optional CSV dump (warning: huge!)
    # pd.read_parquet(OUT_FULL_PQ).to_csv(OUT_FULL_CSV, index=False)
else:
    print("  No files were written?")
