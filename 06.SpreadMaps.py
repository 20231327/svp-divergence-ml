"""
06.SpreadMaps.py
Creates high‑resolution, publication‑ready graphics
Basin x Depth heat‑maps  (mean & p95 × 4 seasons)
2×2 overview heat‑maps  (mean & p95)
Depth‑profiles per basin  (mean & p95)
Clean summary tables (wide) in data/results/ (mean & p95)
"""

# ── imports ────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse  as arg
import pathlib   as pl
import warnings  as wrn
import textwrap  as tw
import pandas    as pd
import numpy      as np
import matplotlib.pyplot  as _plt
import matplotlib         as _mpl
import seaborn            as _sns

# ── plotting defaults ─────────────────────────────────────────────────────
_mpl.rcParams.update({
    "figure.dpi":          110,
    "savefig.dpi":         300,
    "font.family":         "DejaVu Sans",  # unicode‑safe
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.titleweight":    "semibold",
    "axes.titlesize":      14,
    "axes.labelsize":      12,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
})
_sns.set_theme(style="ticks", context="paper")

# colour‑maps
HEAT_CMAP   = _sns.color_palette("crest", as_cmap=True)       # mean  (cool→warm)
HEAT_CMAP_R = _sns.color_palette("rocket_r", as_cmap=True)    # p95   (warm→cool)

# ── I/O paths & CLI ───────────────────────────────────────────────────────
ROOT        = _pl.Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
FIG_DIR     = RESULTS_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

parser = _arg.ArgumentParser(description="Generate SVP‑spread summary plots (NaN‑free)")
parser.add_argument(
    "--csv", dest="csv", type=_pl.Path,
    default=RESULTS_DIR / "spread_summary_by_basin_depth_season.csv",
    help="Path to cleaned summary CSV (or raw multi‑row CSV)"
)
parser.add_argument("--annot", action="store_true", help="Annotate heat‑maps with values")
parser.add_argument("--keep-nan", action="store_true", help="Retain 'nan' depth‑bin column")
args, _ = parser.parse_known_args()
CSV_FP: _pl.Path = args.csv.expanduser().resolve()

if not CSV_FP.exists():
    alt = DATA_DIR / CSV_FP.name
    if alt.exists():
        _wrn.warn(f"Summary CSV not at expected path, using fallback → {alt.relative_to(ROOT)}")
        CSV_FP = alt
    else:
        raise FileNotFoundError(_tw.dedent(f"""
            Summary CSV not found.
            Expected at {CSV_FP.relative_to(ROOT)} or {alt.relative_to(ROOT)}.  
            Run 05C.AggregateStats first (or point --csv to the file you sent in chat).
        """))
print(f"→ reading aggregated spread summaries → {CSV_FP.relative_to(ROOT)}")

# ── helper: robust loader ─────────────────────────────────────────────────
SEASONS          = ["DJF", "MAM", "JJA", "SON"]
_PHYSICAL_DEPTHS = ["0–100", "100–300", "300–700", "700–1200", "1200–1750"]
BASIN_ORDER      = [
    "Arctic Ocean", "N. Atlantic", "S. Atlantic", "Indian Ocean",
    "E. Pacific", "W. Pacific", "Southern Ocean",
]


def _load_summary(fp: _pl.Path) -> _pd.DataFrame:
    """Load either tidy CSV (single header) or raw multi‑row CSV."""
    df = _pd.read_csv(fp)
    if "depth_bin" in df.columns:
        return df
    # raw version: first two rows are helper rows – skip them
    df = _pd.read_csv(fp, skiprows=2)
    cols = df.columns.tolist()
    df.rename(columns={cols[0]: "basin", cols[1]: "depth_bin"}, inplace=True)
    for i, s in enumerate(range(4)):
        df.rename(columns={cols[2+i]:              f"svp_spread_max_mean_{s}"}, inplace=True)
        df.rename(columns={cols[2+4+i]:            f"svp_spread_max_p95_{s}"},  inplace=True)
    return df

summary = _load_summary(CSV_FP)

# clean depth labels & categories
depths   = _pd.Series(summary["depth_bin"].astype(str))
depths   = depths.str.replace("â€“", "–", regex=False).fillna("nan")
summary["depth_bin"] = depths

# drop NaN depth rows unless user explicitly keeps them
if not args.keep_nan:
    summary = summary[summary["depth_bin"] != "nan"].copy()

DEPTH_ORDER = _PHYSICAL_DEPTHS + (["nan"] if args.keep_nan else [])
summary["depth_bin"] = _pd.Categorical(summary["depth_bin"], DEPTH_ORDER, ordered=True)
summary["basin"]     = _pd.Categorical(summary["basin"], BASIN_ORDER, ordered=True)

# ── tidy melt ─────────────────────────────────────────────────────────────
value_cols = [c for c in summary.columns if "spread" in c]
summary = summary.melt(
    id_vars=["basin", "depth_bin"],
    value_vars=value_cols,
    var_name="metric_stat_season",
    value_name="value",
)
parts = summary["metric_stat_season"].str.extract(r"(?P<metric>[^_]+)_(?P<stat>[^_]+)_(?P<season>\d)")
summary = _pd.concat([summary.drop(columns="metric_stat_season"), parts], axis=1)
summary["season"] = summary["season"].astype("uint8").map(dict(enumerate(SEASONS))).astype("category")

# ── export cleaned summary tables (wide) ─────────────────────────────────-
for stat in ("mean", "p95"):
    out_fp = RESULTS_DIR / f"spread_{stat}_basin_depth_season.csv"
    (
        summary.query("stat == @stat")
        .pivot_table(index=["basin", "depth_bin"], columns="season", values="value")
        .to_csv(out_fp)
    )
    print(f" exported {stat} table → {out_fp.relative_to(ROOT)}")

# ── colours: dynamic contrast per‑stat (5–95 percentile) ──────────────────
vlims = (
    summary.groupby("stat")["value"]
    .quantile([0.05, 0.95])
    .unstack(level=1)
    .rename(columns={0.05: "vmin", 0.95: "vmax"})
    .to_dict(orient="index")
)

# list used later for re‑indexing, sans "nan" unless kept
DEPTH_ORDER_FINAL = DEPTH_ORDER

# ── HEAT‑MAPS per season ─────────────────────────────────────────────────
print(" drawing basin–depth heat‑maps per season …")
for stat, cmap in (("mean", HEAT_CMAP), ("p95", HEAT_CMAP_R)):
    vmin, vmax = vlims[stat]["vmin"], vlims[stat]["vmax"]
    for season in SEASONS:
        data = (
            summary.query("stat == @stat and season == @season")
            .pivot(index="basin", columns="depth_bin", values="value")
            .reindex(index=BASIN_ORDER, columns=DEPTH_ORDER_FINAL)
        )
        fig, ax = _plt.subplots(figsize=(8, 5))
        _sns.heatmap(
            data, cmap=cmap, vmin=vmin, vmax=vmax,
            annot=args.annot, fmt=".1f", annot_kws={"size":8},
            linewidths=0.4, linecolor="#f0f0f0",
            cbar_kws={"label": f"Spread {stat} (m)"}, ax=ax,
        )
        ax.set_title(f"{season} – SVP spread (max) {stat}")
        ax.set_xlabel("Depth bin (m)")
        ax.set_ylabel("Basin")
        fig.tight_layout()
        out_fp = FIG_DIR / f"heatmap_basin_depth_{stat}_{season}.png"
        fig.savefig(out_fp, bbox_inches="tight")
        _plt.close(fig)
        print(f"  • {out_fp.relative_to(ROOT)}")

# ── 2×2 OVERVIEWs ────────────────────────────────────────────────────────
print(" drawing 2×2 season overviews …")
for stat, cmap in (("mean", HEAT_CMAP), ("p95", HEAT_CMAP_R)):
    vmin, vmax = vlims[stat]["vmin"], vlims[stat]["vmax"]
    fig, axes = _plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, season in zip(axes.flat, SEASONS):
        data = (
            summary.query("stat == @stat and season == @season")
            .pivot(index="basin", columns="depth_bin", values="value")
            .reindex(index=BASIN_ORDER, columns=DEPTH_ORDER_FINAL)
        )
        _sns.heatmap(
            data, cmap=cmap, vmin=vmin, vmax=vmax,
            annot=args.annot, fmt=".1f", annot_kws={"size":6},
            cbar=False, linewidths=0.3, linecolor="#f0f0f0", ax=ax,
        )
        ax.set_title(season)
    axes[0, 0].set_ylabel("Basin")
    axes[1, 0].set_ylabel("Basin")
    fig.colorbar(
        _plt.cm.ScalarMappable(cmap=cmap, norm=_mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
        ax=axes, label=f"Spread {stat} (m)", fraction=0.03, pad=0.02
    )
    fig.suptitle(f"SVP spread (max) – {stat}")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_fp = FIG_DIR / f"heatmap_basin_depth_{stat}_overview.png"
    fig.savefig(out_fp, bbox_inches="tight")
    _plt.close(fig)
    print(f"  • {out_fp.relative_to(ROOT)}")

# ── Depth‑profiles ───────────────────────────────────────────────────────
print(" drawing depth‑profiles per basin …")
for stat, cmap in (("mean", "mako"), ("p95", "light:#702963")):
    palette = _sns.color_palette(cmap, n_colors=len(SEASONS))
    y_min, y_max = vlims[stat]["vmin"], vlims[stat]["vmax"]
    for basin in BASIN_ORDER:
        data = (
            summary.query("stat == @stat and basin == @basin")
            .pivot(index="depth_bin", columns="season", values="value")
            .reindex(index=DEPTH_ORDER_FINAL)
        )
        fig, ax = _plt.subplots(figsize=(6.5, 4.2))
        for i, season in enumerate(SEASONS):
            ax.plot(data.index, data[season], marker="o", color=palette[i], label=season)
        ax.set_title(f"{basin} – SVP spread {stat}")
        ax.set_xlabel("Depth bin (m)")
        ax.set_ylabel("Spread (m)")
        ax.set_ylim(y_min * 0.98, y_max * 1.02)
        ax.legend(title="Season", fontsize=9)
        fig.tight_layout()
        out_fp = FIG_DIR / f"depth_profile_{stat}_{basin.replace(' ', '_')}.png"
        fig.savefig(out_fp, bbox_inches="tight")
        _plt.close(fig)
        print(f"  • {out_fp.relative_to(ROOT)}")

print(f"\n all figures written → {FIG_DIR.relative_to(ROOT)}")
