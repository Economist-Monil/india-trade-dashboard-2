"""
tariff_analysis.py
Joins CBIC tariff data onto trade flows and produces 4 analytics parquets.
Run AFTER build_tariff_parquets.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

# ── Load base parquets ────────────────────────────────────────────────────────
print("Loading base parquets…")
tariff_hs8 = pd.read_parquet(PROC / "tariff_hs8.parquet")
fact       = pd.read_parquet(PROC / "fact_trade_flows.parquet")
tariff_hs4 = pd.read_parquet(PROC / "tariff_hs4_agg.parquet")
china_dep  = pd.read_parquet(PROC / "china_dependence.parquet")
dim_prod   = pd.read_parquet(PROC / "dim_product.parquet")
ids_hs4    = pd.read_parquet(PROC / "ids_signals.parquet")

print(f"  tariff_hs8       : {len(tariff_hs8):,} rows")
print(f"  fact_trade_flows : {len(fact):,} rows")
print(f"  tariff_hs4_agg   : {len(tariff_hs4):,} rows")
print(f"  dim_product      : {len(dim_prod):,} rows")


# ── Step 0: tariff_hs6_agg.parquet ───────────────────────────────────────────
print("\nStep 0 — building tariff_hs6_agg…")

tariff_hs8 = tariff_hs8.copy()
tariff_hs8["hs6"] = tariff_hs8["hs8"].str[:6]

hs6_agg = (
    tariff_hs8.groupby("hs6")
    .agg(
        bcd_avg       = ("bcd_applied",    "mean"),
        igst_avg      = ("igst_rate",      "mean"),
        total_eff_avg = ("total_eff_duty", "mean"),
        n_lines       = ("hs8",            "count"),
    )
    .reset_index()
)
hs6_agg["hs4"] = hs6_agg["hs6"].str[:4]
hs6_agg["hs2"] = hs6_agg["hs6"].str[:2]

hs6_agg = hs6_agg.loc[:, ~hs6_agg.columns.duplicated()]
hs6_agg.to_parquet(PROC / "tariff_hs6_agg.parquet", index=False)
n_hs85_hs6 = (hs6_agg["hs2"] == "85").sum()
print(f"  tariff_hs6_agg.parquet : {len(hs6_agg):,} rows total | {n_hs85_hs6} in HS85")


# ── Step 1: Enrich fact with tariff rates ─────────────────────────────────────
print("\nStep 1 — joining tariff rates onto trade flows…")
fact_enriched = fact.merge(
    tariff_hs4[["hs4", "bcd_avg", "igst_avg", "total_eff_avg"]],
    on="hs4",
    how="left",
)
match_pct = fact_enriched["bcd_avg"].notna().mean() * 100
print(f"  fact_enriched : {len(fact_enriched):,} rows | tariff match: {match_pct:.1f}%")


# ── Step 2: tariff_profile_hs4.parquet ───────────────────────────────────────
print("\nStep 2 — building tariff_profile_hs4…")

hs85_imp = fact_enriched[
    (fact_enriched["hs2"] == "85") & (fact_enriched["flow"] == "M")
].copy()

profile = (
    hs85_imp.groupby("hs4")
    .agg(
        import_value_bn = ("value_usd",    lambda x: x.sum() / 1e9),
        bcd_avg         = ("bcd_avg",       "first"),
        igst_avg        = ("igst_avg",      "first"),
        total_eff_avg   = ("total_eff_avg", "first"),
    )
    .reset_index()
)

profile = profile.merge(dim_prod[["hs4", "good_type"]], on="hs4", how="left")
profile["good_type"] = profile["good_type"].fillna("UNCLASSIFIED")

profile = profile.merge(china_dep[["hs4", "china_share", "risk_level"]], on="hs4", how="left")
profile["china_share"] = profile["china_share"].fillna(0.0)
profile["risk_level"]  = profile["risk_level"].fillna("LOW")

profile = profile.merge(ids_hs4[["hs4", "ids_signal", "ids_gap_pp"]], on="hs4", how="left")
profile["ids_signal"] = profile["ids_signal"].map(lambda x: bool(x) if pd.notna(x) else False)
profile["ids_gap_pp"] = profile["ids_gap_pp"].fillna(0.0)

profile = (profile.sort_values("import_value_bn", ascending=False)
                  .reset_index(drop=True))
profile = profile.loc[:, ~profile.columns.duplicated()]
profile.to_parquet(PROC / "tariff_profile_hs4.parquet", index=False)
print(f"  tariff_profile_hs4.parquet : {len(profile):,} rows")


# ── Step 3: inverted_duty_analysis.parquet — HS6 LEVEL ───────────────────────
print("\nStep 3 — building inverted_duty_analysis at HS6…")

# HS85 HS6 tariff lines + good_type inherited from parent HS4
hs85_hs6 = hs6_agg[hs6_agg["hs2"] == "85"].copy()
hs85_hs6 = hs85_hs6.merge(dim_prod[["hs4", "good_type", "hs4_desc"]], on="hs4", how="left")
hs85_hs6["good_type"] = hs85_hs6["good_type"].fillna("UNCLASSIFIED")
hs85_hs6["hs4_desc"]  = hs85_hs6["hs4_desc"].fillna("HS" + hs85_hs6["hs4"].astype(str))

# Label each HS6 as INPUT (intermediate/capital) or OUTPUT (final)
INPUT_TYPES = {"INTERMEDIATE", "CAPITAL"}
hs85_hs6["input_output_label"] = hs85_hs6["good_type"].map(
    lambda g: "INPUT" if g in INPUT_TYPES else ("OUTPUT" if g == "FINAL" else "OTHER")
)

# Reference BCD = unweighted mean of all OUTPUT HS6 lines in HS85
output_rows = hs85_hs6[hs85_hs6["input_output_label"] == "OUTPUT"]
output_ref_bcd = float(output_rows["bcd_avg"].mean()) if not output_rows.empty else 0.0
print(f"  OUTPUT HS6 reference avg BCD : {output_ref_bcd:.2f}%")

# ids_gap_pp: for INPUT rows = this line's BCD − output_ref_bcd; for others = 0
hs85_hs6["ids_gap_pp"] = 0.0
mask_in = hs85_hs6["input_output_label"] == "INPUT"
hs85_hs6.loc[mask_in, "ids_gap_pp"] = hs85_hs6.loc[mask_in, "bcd_avg"] - output_ref_bcd
hs85_hs6["ids_flag"] = hs85_hs6["ids_gap_pp"] > 0

# ── HS6 import values from trade data ────────────────────────────────────────
hs85_imports = fact[(fact["hs2"] == "85") & (fact["flow"] == "M")].copy()

imp_hs6 = (
    hs85_imports.groupby("hs6")["value_usd"]
    .sum().reset_index()
)
imp_hs6["import_value_bn"] = imp_hs6["value_usd"] / 1e9

hs85_hs6 = hs85_hs6.merge(imp_hs6[["hs6", "import_value_bn"]], on="hs6", how="left")
hs85_hs6["import_value_bn"] = hs85_hs6["import_value_bn"].fillna(0.0)

# ── China share at HS6 level ──────────────────────────────────────────────────
total_hs6 = hs85_imports.groupby("hs6")["value_usd"].sum().rename("total_usd")
china_hs6 = (
    hs85_imports[hs85_imports["partner_iso3"].astype(str) == "156"]
    .groupby("hs6")["value_usd"].sum().rename("china_usd")
)
china_df = pd.concat([total_hs6, china_hs6], axis=1).reset_index().fillna(0)
china_df["china_share"] = (
    china_df["china_usd"] / china_df["total_usd"].replace(0, float("nan"))
).fillna(0.0)

hs85_hs6 = hs85_hs6.merge(china_df[["hs6", "china_share"]], on="hs6", how="left")
hs85_hs6["china_share"] = hs85_hs6["china_share"].fillna(0.0)

# ── Risk level at HS6 level ───────────────────────────────────────────────────
def risk_fn(row):
    s, lbl = row["china_share"], row["input_output_label"]
    if s > 0.7 and lbl == "INPUT": return "CRITICAL"
    if s > 0.5 and lbl == "INPUT": return "HIGH"
    if s > 0.3: return "MEDIUM"
    return "LOW"

hs85_hs6["risk_level"] = hs85_hs6.apply(risk_fn, axis=1)

# ── Final output ──────────────────────────────────────────────────────────────
ida = hs85_hs6[[
    "hs6", "hs4", "hs4_desc", "good_type", "input_output_label",
    "bcd_avg", "import_value_bn",
    "ids_gap_pp", "ids_flag",
    "china_share", "risk_level",
]].sort_values("ids_gap_pp", ascending=False).reset_index(drop=True)

ida = ida.loc[:, ~ida.columns.duplicated()]
ida.to_parquet(PROC / "inverted_duty_analysis.parquet", index=False)
print(f"  inverted_duty_analysis.parquet : {len(ida):,} rows")
print(f"  IDS-flagged INPUT lines        : {int(ida['ids_flag'].sum())}")


# ── Step 4: tariff_by_good_type.parquet ──────────────────────────────────────
print("\nStep 4 — building tariff_by_good_type…")

def weighted_avg(grp, val_col, wt_col):
    wt    = grp[wt_col].fillna(0)
    denom = wt.sum()
    return float((grp[val_col].fillna(0) * wt).sum() / denom) if denom > 0 else float(grp[val_col].mean())

rows = []
for gtype, grp in profile[profile["good_type"] != "UNCLASSIFIED"].groupby("good_type"):
    rows.append({
        "good_type":       gtype,
        "avg_bcd":         weighted_avg(grp, "bcd_avg",       "import_value_bn"),
        "avg_total_eff":   weighted_avg(grp, "total_eff_avg", "import_value_bn"),
        "import_value_bn": grp["import_value_bn"].sum(),
    })

gtype_df = (
    pd.DataFrame(rows)
    .sort_values("import_value_bn", ascending=False)
    .reset_index(drop=True)
)
gtype_df = gtype_df.loc[:, ~gtype_df.columns.duplicated()]
gtype_df.to_parquet(PROC / "tariff_by_good_type.parquet", index=False)
print(f"  tariff_by_good_type.parquet : {len(gtype_df):,} rows")


# ── Validation ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TARIFF ANALYSIS VALIDATION")
print("=" * 60)

print("\n  tariff_profile_hs4 (top 6 by import value):")
print(profile[["hs4", "good_type", "import_value_bn", "bcd_avg",
               "ids_signal", "risk_level"]].head(6).to_string(index=False))

print(f"\n  inverted_duty_analysis — INPUT type distribution:")
print(ida["input_output_label"].value_counts().to_string())
print(f"\n  Top 10 IDS-flagged INPUT HS6 lines:")
print(ida[ida["ids_flag"]][["hs6", "hs4_desc", "bcd_avg",
                              "ids_gap_pp", "import_value_bn"]].head(10).to_string(index=False))

print(f"\n  OUTPUT reference BCD : {output_ref_bcd:.2f}%")

print("\n  tariff_by_good_type:")
print(gtype_df.to_string(index=False))

inter = gtype_df[gtype_df["good_type"] == "INTERMEDIATE"]
final = gtype_df[gtype_df["good_type"] == "FINAL"]
if not inter.empty and not final.empty:
    penalty = float(inter["avg_bcd"].iloc[0]) - float(final["avg_bcd"].iloc[0])
    print(f"\n  Input Tax Penalty (INTERMEDIATE - FINAL): {penalty:+.2f} pp")
print("=" * 60)
