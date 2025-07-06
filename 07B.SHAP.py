"""
07B.SHAP.py  – RF + TreeSHAP
Retrain the two best RFs (core & augmented) on train+val
Compute TreeSHAP on the held-out test split
Aggregate SHAP globally, by ocean-basin & by depth-bin×season
Export CSVs + publication-ready figures
"""
from pathlib import Path
import datetime as dt
import joblib, json, warnings
import numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------#
# 1  Paths & constants
# ------------------------------------------------------------------#
ROOT        = Path(__file__).resolve().parents[1]
DATA        = ROOT / "data"
RESULT_FIGS = ROOT / "results" / "figs";    RESULT_FIGS.mkdir(parents=True, exist_ok=True)
MODEL_DIR   = ROOT / "models";              MODEL_DIR.mkdir(exist_ok=True)

TR_RAW = DATA / "armor3d_train_raw.parquet"
VA_RAW = DATA / "armor3d_val_raw.parquet"
TE_RAW = DATA / "armor3d_test_raw.parquet"
MASKF  = DATA / "ocean_mask_1deg.npz"

FEATURES_CORE = ["to", "so", "depth_val"]
FEATURES_AUG  = FEATURES_CORE + ["speed", "sin_dir", "cos_dir", "log_mld"]
LABEL         = "svp_spread_max"

BASIN_ORDER = [
    "Arctic Ocean", "N. Atlantic", "S. Atlantic",
    "Indian Ocean", "E. Pacific", "W. Pacific",
    "Southern Ocean"
]

# ------------------------------------------------------------------#
# 2  Load splits  &  add basin labels (vectorised)
# ------------------------------------------------------------------#
print(" Reading splits …")
tr = pd.read_parquet(TR_RAW)
va = pd.read_parquet(VA_RAW)
te = pd.read_parquet(TE_RAW)
print(f"   rows  train {len(tr):,} | val {len(va):,} | test {len(te):,}")

# --- vectorised basin lookup --------------------------------------
npz  = np.load(MASKF)
MASK = npz["mask"];  LAT0 = npz["lat0"];  LON0 = npz["lon0"]

id2label = {1: "Arctic Ocean", 2: "N. Atlantic", 3: "S. Atlantic",
            4: "Indian Ocean", 5: "E. Pacific", 6: "W. Pacific",
            7: "Southern Ocean"}

def add_basin(df: pd.DataFrame) -> pd.DataFrame:
    li = np.floor(df["latitude"].values  - LAT0 + 0.5).astype("int16")
    lo = np.floor((df["longitude"].values - LON0 + 0.5) % 360).astype("int16")
    codes  = MASK[li, lo]
    df["basin"] = pd.Series(codes).map(id2label).fillna("Unknown").to_numpy()
    return df

tr = add_basin(tr);  va = add_basin(va);  te = add_basin(te)

# ------------------------------------------------------------------#
# 3  Fit final Random-Forests  (core & augmented)
# ------------------------------------------------------------------#
def fit_rf(tag: str, X, y, params: dict) -> RandomForestRegressor:
    rf = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, MODEL_DIR / f"{tag}.joblib")
    with open(MODEL_DIR / f"{tag}.json", "w") as f:
        json.dump({"tag": tag,
                   "params": params,
                   "train_rows": len(X),
                   "timestamp": dt.datetime.utcnow().isoformat()}, f, indent=2)
    return rf

rf_params = {"n_estimators": 300, "max_depth": None, "min_samples_split": 2}

print("\n Training final RFs …")
X_tv_core = pd.concat([tr[FEATURES_CORE], va[FEATURES_CORE]], ignore_index=True)
y_tv      = pd.concat([tr[LABEL],         va[LABEL]],         ignore_index=True)
rf_core   = fit_rf("RF_core_final", X_tv_core, y_tv, rf_params)

X_tv_aug  = pd.concat([tr[FEATURES_AUG], va[FEATURES_AUG]], ignore_index=True)
rf_aug    = fit_rf("RF_aug_final",  X_tv_aug,  y_tv, rf_params)
print("    models saved to models/")

# ------------------------------------------------------------------#
# 4  TreeSHAP on test set
# ------------------------------------------------------------------#
def shap_on_test(rf: RandomForestRegressor, X_test: pd.DataFrame,
                 tag: str) -> np.ndarray:
    exp = shap.TreeExplainer(rf, feature_names=X_test.columns.tolist())
    sv  = exp.shap_values(X_test)
    np.savez_compressed(DATA / f"shap_values_test_{tag}.npz", shap_values=sv)
    print(f"   ✓ SHAP array ({tag}) saved")
    return sv

print("\n SHAP on test set …")
sv_core = shap_on_test(rf_core, te[FEATURES_CORE], "core")
sv_aug  = shap_on_test(rf_aug,  te[FEATURES_AUG],  "aug")

# ------------------------------------------------------------------#
# 5  Global importance plots (core model shown in thesis)
# ------------------------------------------------------------------#
def plot_global_bar(shap_vals, X, tag, palette="steelblue"):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    plt.figure(figsize=(6,4))
    sns.barplot(x=mean_abs[order], y=np.array(X.columns)[order],
                color=palette)
    plt.xlabel("mean |SHAP| (m)")
    plt.title(f"Global feature importance – RF {tag}")
    plt.tight_layout()
    plt.savefig(RESULT_FIGS / f"shap_summary_bar_{tag}.png", dpi=300)
    plt.close()

plot_global_bar(sv_core, te[FEATURES_CORE], "core")
plot_global_bar(sv_aug,  te[FEATURES_AUG],  "aug", palette="#AD1457")
print("   shap_summary_bar_* written")

# Beeswarm (core only, for manuscript)
plt.figure(figsize=(7,5))
shap.summary_plot(sv_core, te[FEATURES_CORE], feature_names=FEATURES_CORE,
                  show=False, plot_type="dot", color_bar=True)
plt.tight_layout()
plt.savefig(RESULT_FIGS / "shap_beeswarm_core.png", dpi=300)
plt.close()
print("    shap_beeswarm_core.png written")

# ------------------------------------------------------------------#
# 6  Regional aggregation  (per-basin |SHAP|)
# ------------------------------------------------------------------#
shap_core_df = pd.DataFrame(sv_core, columns=FEATURES_CORE)
agg_basin = (
    shap_core_df.abs()
    .assign(basin=te["basin"].values)
    .groupby("basin")[FEATURES_CORE].mean()
    .reindex(index=BASIN_ORDER)
)
agg_basin.to_csv(DATA / "shap_mean_abs_basin.csv")
plt.figure(figsize=(7,4))
sns.heatmap(agg_basin, cmap="crest", annot=True, fmt=".3f")
plt.title("Mean |SHAP| by basin – RF core")
plt.tight_layout()
plt.savefig(RESULT_FIGS / "shap_basin_heatmap.png", dpi=300)
plt.close()
print("    shap_basin_heatmap.png written")

# ------------------------------------------------------------------#
# 7  Case-study waterfall (highest spread)
# ------------------------------------------------------------------#
shap_df = shap_core_df.assign(spread=te[LABEL].values)
idx_top = shap_df["spread"].idxmax()
shap.force_plot(
    shap.TreeExplainer(rf_core).expected_value,
    sv_core[idx_top, :],
    matplotlib=True,
    feature_names=FEATURES_CORE
)
plt.savefig(RESULT_FIGS / f"shap_breakdown_case{idx_top}.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("    shap_breakdown_case*.png written")

# ------------------------------------------------------------------#
# 8  Global CSVs
# ------------------------------------------------------------------#
pd.DataFrame({
    "feature": FEATURES_CORE,
    "mean_abs_shap": np.abs(sv_core).mean(axis=0)
}).to_csv(DATA / "shap_mean_abs_global.csv", index=False)

print("\n SHAP analysis done – artefacts in data/ and results/figs/")
print("   timestamp:", dt.datetime.utcnow().isoformat(" ", 'seconds'))
