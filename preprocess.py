"""
preprocess.py
=============
Robust preprocessing for India trade data.
Detects columns defensively, builds canonical schema, never crashes.

Canonical output schema:
    year            int
    flow            str  ('X' or 'M')
    reporter_iso3   str
    partner_iso3    str
    partner_name    str
    product_code    str  (zero-padded, no decimals, 6 digits from HS6 data)
    product_desc    str
    value_usd       float
    qty_kg          float
    hs2             str  (2-digit, always present)
    hs4             str  (4-digit)
    hs6             str  (6-digit)
    hs8             str  (8-digit, only from tariff-line data)
    hs2_label       str
    good_type       str
    hs_revision     str
    data_source     str
"""

import pandas as pd
from pathlib import Path

# ── Column detection aliases ──────────────────────────────────────────

PRODUCT_CODE_ALIASES = [
    "hs_code","product_code","hs6","hs8","cmdCode","CmdCode",
    "cmd_code","hscode","ProductCode","product","cmd",
]
PRODUCT_DESC_ALIASES = [
    "product_desc","cmdDesc","CmdDesc","cmd_desc","description",
    "ProductDesc","product_description","desc",
]
VALUE_ALIASES = [
    "value_usd","primaryValue","PrimaryValue","fob_value_usd",
    "FOBValue","FOBvalue","TradeValue","tradeValue","value",
    "trade_value","trade_value_usd",
]
FLOW_ALIASES   = ["flow","flowCode","FlowCode","flow_code","TradeFlow"]
YEAR_ALIASES   = ["year","period","Period","refYear","RefYear","Year"]
PARTNER_ISO_ALIASES  = ["partner_iso3","partnerISO","PartnerISO","partner_iso",
                        "partner_code","partnerCode","PartnerCode"]
PARTNER_NAME_ALIASES = ["partner_name","partnerDesc","PartnerDesc","partner",
                        "PartnerName"]
REPORTER_ISO_ALIASES = ["reporter_iso3","reporterISO","ReporterISO","reporter_iso",
                        "reporter_code","reporterCode","ReporterCode"]
QTY_ALIASES    = ["qty_kg","netWgt","NetWgt","net_weight","qty","weight_kg"]

# ── Reference data ────────────────────────────────────────────────────

HS2_LABELS = {
    "01":"Live animals","02":"Meat & offal","03":"Fish & seafood",
    "04":"Dairy & eggs","05":"Animal products NES","06":"Live plants",
    "07":"Vegetables","08":"Fruits & nuts","09":"Coffee & spices",
    "10":"Cereals","11":"Milling products","12":"Oil seeds",
    "13":"Lac gums resins","14":"Vegetable plaiting","15":"Fats & oils",
    "16":"Meat preparations","17":"Sugars","18":"Cocoa",
    "19":"Cereal preparations","20":"Vegetable preparations",
    "21":"Misc food","22":"Beverages","23":"Animal feed","24":"Tobacco",
    "25":"Salt sulphur stone","26":"Ores & slag","27":"Mineral fuels",
    "28":"Inorganic chemicals","29":"Organic chemicals",
    "30":"Pharmaceuticals","31":"Fertilizers","32":"Dyes & pigments",
    "33":"Perfumes & cosmetics","34":"Soap & detergents",
    "35":"Albuminoids","36":"Explosives","37":"Photographic goods",
    "38":"Chemical products NES","39":"Plastics","40":"Rubber",
    "41":"Hides & leather","42":"Leather articles","43":"Furskins",
    "44":"Wood","45":"Cork","46":"Basketwork","47":"Wood pulp",
    "48":"Paper & paperboard","49":"Printed books",
    "50":"Silk","51":"Wool","52":"Cotton","53":"Vegetable fibres",
    "54":"Man-made filaments","55":"Man-made staple fibres",
    "56":"Wadding & felt","57":"Carpets","58":"Special fabrics",
    "59":"Coated textiles","60":"Knitted fabric",
    "61":"Knitted apparel","62":"Woven apparel",
    "63":"Textile articles NES","64":"Footwear","65":"Headgear",
    "66":"Umbrellas","67":"Feathers","68":"Stone articles",
    "69":"Ceramics","70":"Glass","71":"Gems & precious metals",
    "72":"Iron & steel","73":"Iron & steel articles","74":"Copper",
    "75":"Nickel","76":"Aluminium","78":"Lead","79":"Zinc","80":"Tin",
    "81":"Other base metals","82":"Tools & cutlery",
    "83":"Misc base metal articles","84":"Machinery & mechanical",
    "85":"Electronics & electrical","86":"Railway equipment",
    "87":"Vehicles","88":"Aircraft","89":"Ships",
    "90":"Optical & medical","91":"Clocks & watches",
    "92":"Musical instruments","93":"Arms","94":"Furniture",
    "95":"Toys & games","96":"Misc manufactured","97":"Art & antiques",
}

COUNTRY_NAMES = {
    "156":"China","410":"South Korea","344":"Hong Kong","704":"Vietnam",
    "490":"Other Asia NES","842":"United States","840":"United States",
    "702":"Singapore","392":"Japan","276":"Germany","458":"Malaysia",
    "784":"UAE","528":"Netherlands","826":"United Kingdom","380":"Italy",
    "251":"France","203":"Czechia","643":"Russia","764":"Thailand",
    "158":"Taiwan","682":"Saudi Arabia","36":"Australia","124":"Canada",
    "76":"Brazil","360":"Indonesia","608":"Philippines","724":"Spain",
    "752":"Sweden","756":"Switzerland","710":"South Africa","404":"Kenya",
    "50":"Bangladesh","586":"Pakistan","144":"Sri Lanka","792":"Turkey",
    "818":"Egypt","616":"Poland","372":"Ireland","376":"Israel",
    "414":"Kuwait","634":"Qatar","512":"Oman","699":"India","356":"India",
    "100":"Bulgaria","170":"Colombia","566":"Nigeria","400":"Jordan",
}

FLOW_NORM = {
    "X":"X","M":"M","Export":"X","Import":"M",
    "Re-Export":"X","Re-Import":"M","Re-export":"X","Re-import":"M",
    "1":"M","2":"X","3":"X","4":"M",
}


def _detect(cols, aliases):
    col_set = set(cols)
    for a in aliases:
        if a in col_set:
            return a
    return None


def detect_columns(df):
    cols = df.columns.tolist()
    return {
        "product_code": _detect(cols, PRODUCT_CODE_ALIASES),
        "product_desc": _detect(cols, PRODUCT_DESC_ALIASES),
        "value_usd":    _detect(cols, VALUE_ALIASES),
        "flow":         _detect(cols, FLOW_ALIASES),
        "year":         _detect(cols, YEAR_ALIASES),
        "partner_iso3": _detect(cols, PARTNER_ISO_ALIASES),
        "partner_name": _detect(cols, PARTNER_NAME_ALIASES),
        "reporter_iso3":_detect(cols, REPORTER_ISO_ALIASES),
        "qty_kg":       _detect(cols, QTY_ALIASES),
    }


def build_hs_hierarchy(product_code_series):
    """
    Build hs2/hs4/hs6/hs8 from a cleaned product code series.
    Returns DataFrame. Empty/invalid codes produce pd.NA at each level.
    """
    s = product_code_series.astype(str).str.strip()
    # Remove non-digits
    s = s.str.replace(r"[^0-9]", "", regex=True)
    # Mark invalid (empty, 'nan', <2 digits)
    valid = s.str.len() >= 2
    s = s.where(valid, pd.NA)

    def safe_slice(ser, n):
        """Return first n chars where length >= n, else pd.NA."""
        long_enough = ser.str.len() >= n
        result = ser.str[:n]
        return result.where(long_enough & ser.notna(), pd.NA)

    return pd.DataFrame({
        "hs2": safe_slice(s, 2),
        "hs4": safe_slice(s, 4),
        "hs6": safe_slice(s, 6),
        "hs8": safe_slice(s, 8),
    })


def get_available_levels(df):
    """
    Return HS levels that have real non-null data, ordered finest-first.
    Excludes hs8 unless genuinely populated (>1% of rows).
    """
    total = max(len(df), 1)
    levels = []
    for lv in ["hs8","hs6","hs4","hs2"]:
        if lv in df.columns:
            n = df[lv].notna().sum()
            pct = n / total
            # hs8: only include if at least 1% of rows have it
            # hs6+: include if at least 10% of rows have it
            threshold = 0.01 if lv == "hs8" else 0.10
            if pct >= threshold:
                levels.append(lv)
    return levels if levels else ["hs2"]


def validate(df):
    print("\n" + "="*60)
    print("  DATA VALIDATION REPORT")
    print("="*60)
    print(f"  Shape  : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Columns: {df.columns.tolist()}")
    for col in ["hs2","hs4","hs6","hs8","product_code","value_usd","flow","year"]:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct = 100*n_valid/len(df) if len(df)>0 else 0
            sample = df[col].dropna().head(3).tolist()
            print(f"  {col:<14}: {n_valid:,} valid ({pct:.0f}%) | {sample}")
        else:
            print(f"  {col:<14}: NOT PRESENT")
    if "year" in df.columns:
        print(f"  Years  : {sorted(df['year'].dropna().unique().tolist())}")
    print("="*60)


def safe_nunique(df, prefer):
    for col in prefer:
        if col in df.columns and df[col].notna().any():
            return int(df[col].nunique())
    return 0


def preprocess(df):
    """Full preprocessing. Accepts any column naming, outputs canonical schema."""
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    mapping = detect_columns(df)

    # Rename detected columns → canonical names
    rename = {}
    for canonical, raw in mapping.items():
        if raw and raw != canonical and raw in df.columns:
            # Don't rename if target already exists as a different column
            if canonical not in df.columns:
                rename[raw] = canonical
    if rename:
        df = df.rename(columns=rename)

    # ── Product code ──────────────────────────────────────────────────
    if "product_code" not in df.columns:
        # Try to find any HS column that exists
        for col in ["hs6","hs4","hs2","hs_code"]:
            if col in df.columns:
                df["product_code"] = df[col].astype(str)
                break
        else:
            df["product_code"] = pd.NA

    # ── Build HS hierarchy from product_code ──────────────────────────
    hier = build_hs_hierarchy(df["product_code"].fillna("").astype(str))
    df["hs2"] = hier["hs2"]
    df["hs4"] = hier["hs4"]
    df["hs6"] = hier["hs6"]
    df["hs8"] = hier["hs8"]

    # ── Chapter label ──────────────────────────────────────────────────
    df["hs2_label"] = df["hs2"].map(HS2_LABELS).fillna(
        "HS" + df["hs2"].fillna("??").astype(str)
    )

    # ── Value ──────────────────────────────────────────────────────────
    if "value_usd" not in df.columns:
        for alt in ["fob_value_usd","cif_value_usd"]:
            if alt in df.columns:
                df["value_usd"] = pd.to_numeric(df[alt], errors="coerce")
                break
        else:
            df["value_usd"] = pd.NA
    else:
        df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce")

    # ── Flow ───────────────────────────────────────────────────────────
    if "flow" in df.columns:
        df["flow"] = df["flow"].map(FLOW_NORM).fillna(df["flow"].astype(str))
    else:
        df["flow"] = pd.NA

    # ── Year ───────────────────────────────────────────────────────────
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    else:
        df["year"] = pd.NA

    # ── Partner ────────────────────────────────────────────────────────
    if "partner_iso3" not in df.columns:
        df["partner_iso3"] = "UNK"
    else:
        df["partner_iso3"] = df["partner_iso3"].astype(str).str.strip()

    if "partner_name" not in df.columns or df["partner_name"].isna().all():
        df["partner_name"] = (
            df["partner_iso3"].map(COUNTRY_NAMES)
            .fillna("Country " + df["partner_iso3"].astype(str))
        )

    # ── Good type fallback ─────────────────────────────────────────────
    if "good_type" not in df.columns:
        df["good_type"] = "UNCLASSIFIED"

    # ── Product desc fallback ──────────────────────────────────────────
    if "product_desc" not in df.columns:
        df["product_desc"] = ""

    # ── Qty ────────────────────────────────────────────────────────────
    if "qty_kg" in df.columns:
        df["qty_kg"] = pd.to_numeric(df["qty_kg"], errors="coerce")
    else:
        df["qty_kg"] = pd.NA

    # ── Drop rows without value ────────────────────────────────────────
    df = df[df["value_usd"].notna() & (df["value_usd"] > 0)].copy()

    return df


def load_and_preprocess(source):
    """
    Load parquet and preprocess. Accepts Path, str, or DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        return preprocess(source)
    return preprocess(pd.read_parquet(source))
