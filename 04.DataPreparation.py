"""
# 04_DataPreparation.py
Load the three Parquet splits
Drop sparse rows  (>30% NaNs in core vars)
MICE-impute remaining NaNs   (fit on 1% sample of train)
Feature-engineer: speed, sin_dir, cos_dir, log_mld
Drop mlotst, ugo, vgo (collinear with engineered)
Clip to physical bounds + remove Isolation-Forest outliers
Write *_raw.parquet    (for tree models; 7 predictors)
MinMax-scale → *_scaled.parquet  (for linear/NN; dump imputer+scaler)
"""

from pathlib import Path
import random, joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa:F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# ── Paths ────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_PQ = DATA_DIR / "armor3d_train.parquet"
VAL_PQ   = DATA_DIR / "armor3d_val.parquet"
TEST_PQ  = DATA_DIR / "armor3d_test.parquet"
assert TRAIN_PQ.exists(), f"Missing → {TRAIN_PQ}"

# ── Load splits ──────────────────────────────────────────────────────
train_df = pd.read_parquet(TRAIN_PQ)
val_df   = pd.read_parquet(VAL_PQ)
test_df  = pd.read_parquet(TEST_PQ)
print("Rows  train/val/test →", len(train_df), len(val_df), len(test_df))

# ── Drop too‐sparse rows in core vars ─────────────────────────────────
CORE_VARS = ["to", "so", "depth_val", "ugo", "vgo", "mlotst"]
def drop_sparse(df, cols, thresh=0.30):
    keep = df[cols].isna().mean(axis=1) <= thresh
    return df.loc[keep].reset_index(drop=True)

train_df = drop_sparse(train_df, CORE_VARS)
val_df   = drop_sparse(val_df,   CORE_VARS)
test_df  = drop_sparse(test_df,  CORE_VARS)
print("After drop_sparse →", len(train_df), len(val_df), len(test_df))

# ── MICE imputation (fit on 1% subsample of train) ───────────────────
sample_idx = random.sample(
    range(len(train_df)),
    max(1, int(len(train_df) * 0.01))
)
mice = IterativeImputer(
    max_iter=10,
    sample_posterior=True,
    random_state=42
)
mice.fit(train_df.loc[sample_idx, CORE_VARS])

for df in (train_df, val_df, test_df):
    df[CORE_VARS] = mice.transform(df[CORE_VARS])
print("MICE imputation done")

# ── Feature engineering + drop collinear originals ───────────────────
def add_features(df):
    spd = np.hypot(df["ugo"], df["vgo"])
    df["speed"]   = spd
    dir_safe      = spd.replace(0, np.nan)
    df["sin_dir"] = (df["vgo"] / dir_safe).astype(np.float32)
    df["cos_dir"] = (df["ugo"] / dir_safe).astype(np.float32)
    # log MLD (+1 to avoid log(0))
    df["log_mld"] = np.log10(df["mlotst"].clip(lower=0) + 1)
    return df

for df in (train_df, val_df, test_df):
    add_features(df)

# drop the originals that are now collinear
DROP_VARS  = ["mlotst", "ugo", "vgo"]
for df in (train_df, val_df, test_df):
    df.drop(columns=DROP_VARS, inplace=True)

# final predictor list
PREDICTORS = ["to", "so", "depth_val",
              "speed", "sin_dir", "cos_dir",
              "log_mld"]
TARGETS    = ["svp_mackenzie", "svp_coppens",
              "svp_unesco", "svp_del_grosso", "svp_npl"]

# ── Clip to physical bounds + outlier‐filter ─────────────────────────
bounds = {
    "to":        (-3, 40),   # °C
    "so":        (0, 50),    # PSU
    "depth_val": (0, 8000),  # m
    "speed":     (0, 5),     # m/s
    "sin_dir":   (-1, 1),
    "cos_dir":   (-1, 1),
    "log_mld":   (0, 4),     # sanity cap on log10(MLD+1)
}
def clip_df(df):
    mask = np.ones(len(df), dtype=bool)
    for c,(lo,hi) in bounds.items():
        mask &= df[c].between(lo, hi)
    return df.loc[mask].reset_index(drop=True)

train_df = clip_df(train_df)
val_df   = clip_df(val_df)
test_df  = clip_df(test_df)
print("After clipping →", len(train_df), len(val_df), len(test_df))

iso = IsolationForest(
    contamination=0.02,
    n_estimators=100,
    random_state=42
)
iso.fit(train_df[PREDICTORS])
def keep_inliers(df):
    return df.loc[ iso.predict(df[PREDICTORS]) == 1 ].reset_index(drop=True)

train_df = keep_inliers(train_df)
val_df   = keep_inliers(val_df)
test_df  = keep_inliers(test_df)
print("After outlier filter →", len(train_df), len(val_df), len(test_df))

# ──  Write raw (unscaled) splits for tree‐based models ───────────────
train_df.to_parquet(DATA_DIR / "armor3d_train_raw.parquet", index=False)
val_df  .to_parquet(DATA_DIR / "armor3d_val_raw.parquet",   index=False)
test_df .to_parquet(DATA_DIR / "armor3d_test_raw.parquet",  index=False)
print("✓ wrote *_raw.parquet")

# ──  MinMax‐scale predictors & write scaled splits ──────────────────
scaler = MinMaxScaler()
train_df[PREDICTORS] = scaler.fit_transform(train_df[PREDICTORS])
val_df  [PREDICTORS] = scaler.transform(val_df[  PREDICTORS])
test_df [PREDICTORS] = scaler.transform(test_df [PREDICTORS])

train_df.to_parquet(DATA_DIR / "armor3d_train_scaled.parquet", index=False)
val_df  .to_parquet(DATA_DIR / "armor3d_val_scaled.parquet",   index=False)
test_df .to_parquet(DATA_DIR / "armor3d_test_scaled.parquet",  index=False)

# ──  Persist MICE & scaler ──────────────────────────────────────────
joblib.dump(mice,   DATA_DIR / "mice_imputer.joblib")
joblib.dump(scaler, DATA_DIR / "minmax_scaler.joblib")
print("finished – cleaned, engineered, raw+scaled splits ready")
