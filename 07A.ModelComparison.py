"""
07A.ModelComparison.py
Reads the scaled train/val/test Parquets created in 04_DataPreparation
Benchmarks 4 algorithms (LR, LASSO, RF, tuned MLP) on the SVP‑spread
target *svp_spread_max* under two feature sets:
 core      = [to, so, depth_val]
 augmented = core + [speed, sin_dir, cos_dir, log_mld]
Lightweight hyper‑parameter searches (Grid for LASSO, Random for RF/MLP)
Persists best models + JSON meta under models/
Generates data/model_summary.csv for downstream selection / reporting
"""

# ── 1 ▸ Imports & paths ──────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
import json, time, joblib, warnings
import numpy as np, pandas as pd
from sklearn.linear_model        import LinearRegression, Lasso
from sklearn.ensemble            import RandomForestRegressor
from sklearn.neural_network      import MLPRegressor
from sklearn.model_selection     import GridSearchCV, RandomizedSearchCV
from sklearn.metrics             import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)

TRAIN_PQ   = DATA_DIR / "armor3d_train_scaled.parquet"
VAL_PQ     = DATA_DIR / "armor3d_val_scaled.parquet"
TEST_PQ    = DATA_DIR / "armor3d_test_scaled.parquet"

# ── 2 ▸ Load splits & build feature matrices ─────────────────────────
print("\n🔹 Reading splits …")
train = pd.read_parquet(TRAIN_PQ)
val   = pd.read_parquet(VAL_PQ)
test  = pd.read_parquet(TEST_PQ)
print(f"   train rows: {len(train):,}  | val rows: {len(val):,}  | test rows: {len(test):,}")

aug_cols = ["speed", "sin_dir", "cos_dir", "log_mld"]
CORE_FEATS      = ["to", "so", "depth_val"]
AUGMENTED_FEATS = CORE_FEATS + aug_cols

TARGET = "svp_spread_max"

# helper
def make_xy(df: pd.DataFrame, feats: list[str]):
    X = df[feats].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y

X_train_core,      y_train = make_xy(train, CORE_FEATS)
X_val_core,        y_val   = make_xy(val,   CORE_FEATS)
X_test_core,       y_test  = make_xy(test,  CORE_FEATS)

X_train_aug,  _ = make_xy(train, AUGMENTED_FEATS)
X_val_aug,     _ = make_xy(val,   AUGMENTED_FEATS)
X_test_aug,    _ = make_xy(test,  AUGMENTED_FEATS)

print("   feature matrices built \n")

# ── 3 ▸ Metric helper ────────────────────────────────────────────────
def _mae_rmse(est, X, y):
    p = est.predict(X)
    return mean_absolute_error(y, p), np.sqrt(mean_squared_error(y, p))

# ── 4 ▸ Train → evaluate → save helper ───────────────────────────────

def train_eval_save(tag: str,
                    model,
                    X_train, y_train,
                    X_val,   y_val,
                    X_test,  y_test,
                    param_grid: dict | None = None,
                    use_random_search: bool = False):
    """Fit model (with optional CV search), compute metrics, persist artefacts."""
    tic = time.time()

    # ---- optional hyper‑parameter search ----------------------------
    if param_grid is not None:
        # sample down for speed if >100k rows
        if len(X_train) > 100_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), size=100_000, replace=False)
            X_cv, y_cv = X_train[idx], y_train[idx]
        else:
            X_cv, y_cv = X_train, y_train

        if use_random_search:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=min(20, np.prod([len(v) for v in param_grid.values()])),
                cv=3,
                n_jobs=-1,
                random_state=42,
                scoring="neg_root_mean_squared_error",
                verbose=0,
            )
        else:
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                scoring="neg_root_mean_squared_error",
                verbose=0,
            )
        search.fit(X_cv, y_cv)
        best_est, best_params = search.best_estimator_, search.best_params_
    else:
        model.fit(X_train, y_train)
        best_est, best_params = model, {}

    # ---- metrics ----------------------------------------------------
    mae_v, rmse_v = _mae_rmse(best_est, X_val,  y_val)
    mae_t, rmse_t = _mae_rmse(best_est, X_test, y_test)

    # ---- persist artefacts -----------------------------------------
    joblib.dump(best_est, MODEL_DIR / f"{tag}.joblib")
    (MODEL_DIR / f"{tag}.json").write_text(json.dumps({
        "tag": tag,
        "best_params": best_params,
        "val":  {"MAE": mae_v, "RMSE": rmse_v},
        "test": {"MAE": mae_t, "RMSE": rmse_t},
        "elapsed_s": round(time.time() - tic, 1),
    }, indent=2))

    print(f"{tag:<16s}  val_RMSE={rmse_v:6.3f}  test_RMSE={rmse_t:6.3f}")
    return {"tag": tag, "val_RMSE": rmse_v, "test_RMSE": rmse_t,
            "elapsed_s": round(time.time() - tic, 1)}

# ── 5 ▸ Parameter grids ---------------------------------------------
lasso_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10]}

rf_grid = {
    "n_estimators":      [120, 240, 360],
    "max_depth":         [None, 15, 25],
    "min_samples_split": [2, 4],
}

mlp_grid = {
    "hidden_layer_sizes": [(128, 64), (256, 128, 64)],
    "learning_rate_init": [1e-3, 5e-4],
    "alpha":              [1e-4, 5e-4, 1e-3],
}

# ── 6 ▸ Training loop -----------------------------------------------
results = []

# 6.1 Linear Regression (core & aug)
results.append(
    train_eval_save(
        "LR_core",  LinearRegression(),
        X_train_core, y_train, X_val_core, y_val, X_test_core, y_test,
    )
)
results.append(
    train_eval_save(
        "LR_aug",   LinearRegression(),
        X_train_aug, y_train, X_val_aug,  y_val, X_test_aug,  y_test,
    )
)

# 6.2 LASSO (scaled → core & aug)
results.append(
    train_eval_save(
        "LASSO_core", Lasso(max_iter=20_000, random_state=42),
        X_train_core, y_train, X_val_core, y_val, X_test_core, y_test,
        param_grid=lasso_grid,
    )
)
results.append(
    train_eval_save(
        "LASSO_aug",  Lasso(max_iter=20_000, random_state=42),
        X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test,
        param_grid=lasso_grid,
    )
)

# 6.3 Random Forest (raw values already scaled ~ OK)
results.append(
    train_eval_save(
        "RF_core", RandomForestRegressor(n_jobs=-1, random_state=42),
        X_train_core, y_train, X_val_core, y_val, X_test_core, y_test,
        param_grid=rf_grid, use_random_search=True,
    )
)
results.append(
    train_eval_save(
        "RF_aug", RandomForestRegressor(n_jobs=-1, random_state=42),
        X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test,
        param_grid=rf_grid, use_random_search=True,
    )
)

# 6.4 Tuned MLP (scaled)
print("\n── Tuned‑MLP search (CPU‑friendly) ──")
results.append(
    train_eval_save(
        "MLP_core", MLPRegressor(max_iter=400, solver="adam", learning_rate="adaptive",
                                  batch_size=512, random_state=42, early_stopping=True,
                                  n_iter_no_change=20),
        X_train_core, y_train, X_val_core, y_val, X_test_core, y_test,
        param_grid=mlp_grid, use_random_search=True,
    )
)
results.append(
    train_eval_save(
        "MLP_aug", MLPRegressor(max_iter=400, solver="adam", learning_rate="adaptive",
                                 batch_size=512, random_state=42, early_stopping=True,
                                 n_iter_no_change=20),
        X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test,
        param_grid=mlp_grid, use_random_search=True,
    )
)

# ── 7 ▸ Summary table -------------------------------------------------
summary = pd.DataFrame(results).sort_values("test_RMSE")
summary.to_csv(DATA_DIR / "model_summary.csv", index=False)

print("\nTop models by test RMSE:")
try:
    from tabulate import tabulate
    print(tabulate(summary.head(10), headers="keys", tablefmt="github", floatfmt=".3f"))
except ImportError:
    print(summary.head(10).to_markdown(index=False))

print("\n Model comparison finished – artefacts in models/ and data/model_summary.csv")
