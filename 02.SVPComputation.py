"""
02.SVPComputation.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Sample I/O
sample_in  = DATA_DIR / "armor3d_strat_sample.parquet"
sample_out = DATA_DIR / "armor3d_strat_sample_with_svp.parquet"

# Full I/O
full_in   = DATA_DIR / "armor3d_full.parquet"
full_out  = DATA_DIR / "armor3d_full_with_svp.parquet"

# pressure conversions & SVP funcs (unchanged) …
def depth_to_bar(d):     return d * 0.1
def depth_to_kgcm2(d):   return d * 0.1 * 1.019716

def svp_mackenzie(T, S, D):
    return (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3
            + 1.340*(S-35) + 1.630e-2*D + 1.675e-7*D**2
            - 1.025e-2*T*(S-35) - 7.139e-13*T*D**3)

def svp_coppens(T, S, D_km):
    t = T/10
    return (1449.05 + 45.7*t - 5.21*t**2 + 0.23*t**3
            + (16.23 + 0.253*t)*D_km + (0.213 - 0.1*t)*D_km**2
            + (S-35)*(0.056 + 0.0002*t*D_km))

def svp_unesco(T, S, P_bar):
    C00,C01,C02,C03 = 1402.388, 5.03830, -5.81090e-2, 3.3432e-4
    A00,A01,A02     = 1.389, 1.262e-2, -7.164e-5
    B00,B01         = 9.4742e-5, -1.2580e-5
    D00             = 1.727e-3
    Cw = C00 + C01*T + C02*T**2 + C03*T**3
    A  = A00 + A01*T + A02*T**2
    B  = B00 + B01*T
    return Cw + A*S + B*S**1.5 + D00*S**2 + 1.602e-1*P_bar

def svp_del_grosso(T, S, P_kgcm2):
    C000 = 1402.392
    CT1,CT2,CT3 = 5.01109, -5.462e-2, 1.653e-4
    CS1,CS2     = 1.501e-1, 1.666e-4
    CP1,CP2,CP3 = 1.100e-1, 3.21e-4, 2.0e-6
    return (C000
            + (CT1*T + CT2*T**2 + CT3*T**3)
            + (CS1*S + CS2*S**2)
            + (CP1*P_kgcm2 + CP2*P_kgcm2**2 + CP3*P_kgcm2**3))

def svp_npl(T, S, D, LAT):
    Φ = LAT
    return (1402.5 + 5*T - 5.44e-2*T**2 + 2.1e-4*T**3
            + 1.33*S - 1.23e-2*S*T + 8.7e-5*S*T**2
            + 1.56e-2*D + 2.55e-7*D**2 - 7.3e-12*D**3
            + 1.2e-6*D*(Φ-45) - 9.5e-13*T*D**3
            + 3e-7*T**2*D + 1.43e-5*S*D)

PAIRWISE = [
    ("mackenzie","npl"),
    ("mackenzie","coppens"),
    ("unesco","del_grosso")
]

# 1) SAMPLE
print("=== processing SAMPLE ===")
df = pd.read_parquet(sample_in)
T   = df["to"].to_numpy(float);      S   = df["so"].to_numpy(float)
D   = df["depth_val"].to_numpy(float); LAT = df["latitude"].to_numpy(float)

# compute SVPs & spreads
df["svp_mackenzie"]  = svp_mackenzie(T, S, D).astype(np.float32)
df["svp_coppens"]    = svp_coppens(T, S, D/1000).astype(np.float32)
df["svp_unesco"]     = svp_unesco(T, S, depth_to_bar(D)).astype(np.float32)
df["svp_del_grosso"] = svp_del_grosso(T, S, depth_to_kgcm2(D)).astype(np.float32)
df["svp_npl"]        = svp_npl(T, S, D, LAT).astype(np.float32)

svp_cols = ["svp_mackenzie","svp_coppens","svp_unesco",
            "svp_del_grosso","svp_npl"]
df["svp_spread_max"] = df[svp_cols].max(axis=1) - df[svp_cols].min(axis=1)
df["svp_spread_std"] = df[svp_cols].std(axis=1)
for a,b in PAIRWISE:
    df[f"delta_{a}_{b}"] = (df[f"svp_{a}"] - df[f"svp_{b}"]).abs()

df.to_parquet(sample_out, index=False)
df.to_csv   (sample_out.with_suffix(".csv"), index=False)
print("✓ sample complete")

# 2) FULL (streaming)
print("\n=== processing FULL (streaming) ===")
dataset = ds.dataset(full_in, format="parquet")
scanner = dataset.scanner(batch_size=1_000_000)
writer  = None

for rb in scanner.to_batches():
    pdf = rb.to_pandas()

    T   = pdf["to"].to_numpy(float);      S   = pdf["so"].to_numpy(float)
    D   = pdf["depth_val"].to_numpy(float); LAT = pdf["latitude"].to_numpy(float)

    pdf["svp_mackenzie"]  = svp_mackenzie(T, S, D).astype(np.float32)
    pdf["svp_coppens"]    = svp_coppens(T, S, D/1000).astype(np.float32)
    pdf["svp_unesco"]     = svp_unesco(T, S, depth_to_bar(D)).astype(np.float32)
    pdf["svp_del_grosso"] = svp_del_grosso(T, S, depth_to_kgcm2(D)).astype(np.float32)
    pdf["svp_npl"]        = svp_npl(T, S, D, LAT).astype(np.float32)

    pdf["svp_spread_max"] = pdf[svp_cols].max(axis=1) - pdf[svp_cols].min(axis=1)
    pdf["svp_spread_std"] = pdf[svp_cols].std(axis=1)
    for a,b in PAIRWISE:
        pdf[f"delta_{a}_{b}"] = (pdf[f"svp_{a}"] - pdf[f"svp_{b}"]).abs()

    table = pa.Table.from_pandas(pdf, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(full_out, table.schema)
    writer.write_table(table)

if writer:
    writer.close()
print(" full complete →", full_out)
