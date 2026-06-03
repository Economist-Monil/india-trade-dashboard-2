"""
build_tariff_parquets.py
Reads CBIC_Master_Tariff_2025_26.xlsx and writes 4 parquet files to data/processed/.
Run: python build_tariff_parquets.py
"""
from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
CBIC_PATH = Path(r"S:\Monil_Projects\India_Tariff\output\CBIC_Master_Tariff_2025_26.xlsx")
PROC      = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def zpad(series, width):
    """Zero-pad HS code series to `width` digits. Handles float-read strings."""
    s = (series.astype(str)
               .str.strip()
               .str.replace(r"\.0+$", "", regex=True)   # "8501.0" → "8501"
               .str.replace(r"[^0-9]", "", regex=True)  # drop non-digits
               .str.zfill(width))
    return s


def to_bool(series):
    """Robust bool conversion: handles True/False, 1/0, 'yes'/'no', NaN."""
    s = series.fillna(False).astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y"])


# ── Output 1: tariff_hs8.parquet ──────────────────────────────────────────────
print("Reading HS8 Lines Only…")
df8 = pd.read_excel(
    CBIC_PATH,
    sheet_name="HS8 Lines Only",
    dtype={"HS8": str, "HS4": str, "HS2": str},
)

out8 = pd.DataFrame()
out8["hs8"]            = zpad(df8["HS8"], 8)
out8["hs4"]            = out8["hs8"].str[:4]
out8["hs2"]            = out8["hs8"].str[:2]
out8["product_desc"]   = df8["Product Description"].fillna("").astype(str).str.strip()
out8["bcd_applied"]    = pd.to_numeric(df8["BCD Applied (%)"],          errors="coerce").fillna(0)
out8["igst_rate"]      = pd.to_numeric(df8["IGST Rate (%)"],             errors="coerce").fillna(0)
out8["sws_rate"]       = pd.to_numeric(df8["SWS Rate (%)"],              errors="coerce").fillna(0)
out8["total_eff_duty"] = pd.to_numeric(df8["Total Effective Duty (%)"],  errors="coerce").fillna(0)
out8["import_policy"]  = df8["Import Policy"].fillna("Free").astype(str).str.strip()
out8["has_ntb"]        = to_bool(df8["Has NTB?"])

out8 = out8.loc[:, ~out8.columns.duplicated()]
out8.to_parquet(PROC / "tariff_hs8.parquet", index=False)
print(f"  tariff_hs8.parquet: {len(out8):,} rows")


# ── Output 2: tariff_hs4_agg.parquet ─────────────────────────────────────────
print("Aggregating to HS4…")
agg = (
    out8.groupby(["hs4", "hs2"])
    .agg(
        bcd_avg       = ("bcd_applied",    "mean"),
        bcd_min       = ("bcd_applied",    "min"),
        bcd_max       = ("bcd_applied",    "max"),
        bcd_median    = ("bcd_applied",    "median"),
        igst_avg      = ("igst_rate",      "mean"),
        total_eff_avg = ("total_eff_duty", "mean"),
        n_lines       = ("hs8",            "count"),
    )
    .reset_index()
)

ntb_counts = (
    out8.groupby("hs4")
    .agg(
        restricted_lines = ("import_policy", lambda x: (x != "Free").sum()),
        ntb_lines        = ("has_ntb",       "sum"),
    )
    .reset_index()
)
ntb_counts["ntb_lines"] = ntb_counts["ntb_lines"].astype(int)

agg = agg.merge(ntb_counts, on="hs4", how="left")
agg = agg.loc[:, ~agg.columns.duplicated()]
agg.to_parquet(PROC / "tariff_hs4_agg.parquet", index=False)
print(f"  tariff_hs4_agg.parquet: {len(agg):,} rows")


# ── Output 3: ids_signals.parquet ─────────────────────────────────────────────
print("Reading IDS Signal (HS4)…")
dfi = pd.read_excel(CBIC_PATH, sheet_name="IDS Signal (HS4)")

ids = pd.DataFrame()
ids["hs2"]             = zpad(dfi["HS2 Chapter"],        2)
ids["hs4"]             = zpad(dfi["HS4 Group"],          4)
ids["hs4_avg_bcd"]     = pd.to_numeric(dfi["HS4 Avg BCD (%)"],       errors="coerce").fillna(0)
ids["chapter_avg_bcd"] = pd.to_numeric(dfi["Chapter Avg BCD (%)"],   errors="coerce").fillna(0)
ids["ids_gap_pp"]      = pd.to_numeric(dfi["Gap vs Chapter (pp)"],   errors="coerce").fillna(0)
ids["ids_signal"]      = to_bool(dfi["IDS Signal"])

ids = ids.loc[:, ~ids.columns.duplicated()]
ids.to_parquet(PROC / "ids_signals.parquet", index=False)
print(f"  ids_signals.parquet: {len(ids):,} rows")


# ── Output 4: chapter_summary_tariff.parquet ──────────────────────────────────
print("Reading Chapter Summary…")
dfc = pd.read_excel(CBIC_PATH, sheet_name="Chapter Summary")

cs = pd.DataFrame()
cs["hs2"]             = zpad(dfc["HS2 Chapter"],                    2)
cs["chapter_desc"]    = dfc["Chapter Description (truncated)"].fillna("").astype(str).str.strip()
cs["n_hs8_lines"]     = pd.to_numeric(dfc["HS8 Lines"],                    errors="coerce").fillna(0).astype(int)
cs["bcd_mean"]        = pd.to_numeric(dfc["BCD Mean (%)"],                  errors="coerce").fillna(0)
cs["bcd_median"]      = pd.to_numeric(dfc["BCD Median (%)"],                errors="coerce").fillna(0)
cs["bcd_min"]         = pd.to_numeric(dfc["BCD Min (%)"],                   errors="coerce").fillna(0)
cs["bcd_max"]         = pd.to_numeric(dfc["BCD Max (%)"],                   errors="coerce").fillna(0)
cs["igst_mean"]       = pd.to_numeric(dfc["IGST Mean (%)"],                 errors="coerce").fillna(0)
cs["total_eff_mean"]  = pd.to_numeric(dfc["Total Effective Duty Mean (%)"], errors="coerce").fillna(0)
cs["free_bcd_lines"]  = pd.to_numeric(dfc["Free BCD Lines"],                errors="coerce").fillna(0).astype(int)
cs["ntb_lines"]       = pd.to_numeric(dfc["Lines with NTB"],                errors="coerce").fillna(0).astype(int)
cs["restricted_lines"]= pd.to_numeric(dfc["Restricted Import Lines"],       errors="coerce").fillna(0).astype(int)

cs = cs.loc[:, ~cs.columns.duplicated()]
cs.to_parquet(PROC / "chapter_summary_tariff.parquet", index=False)
print(f"  chapter_summary_tariff.parquet: {len(cs):,} rows")


# ── Validation ─────────────────────────────────────────────────────────────────
hs85      = out8[out8["hs2"] == "85"]
ids_hs85  = ids[ids["hs2"] == "85"]

print("\n" + "=" * 60)
print("  TARIFF PARQUET VALIDATION")
print("=" * 60)
print(f"  tariff_hs8.parquet             : {len(out8):,} rows")
print(f"  tariff_hs4_agg.parquet         : {len(agg):,} rows")
print(f"  ids_signals.parquet            : {len(ids):,} rows")
print(f"  chapter_summary_tariff.parquet : {len(cs):,} rows")
print()
print(f"  HS85 breakdown:")
print(f"    HS8 lines          : {len(hs85):,}")
print(f"    Mean BCD (%)       : {hs85['bcd_applied'].mean():.2f}%")
print(f"    IDS signals        : {int(ids_hs85['ids_signal'].sum())} of {len(ids_hs85)} HS4 groups flagged")
print("=" * 60)
