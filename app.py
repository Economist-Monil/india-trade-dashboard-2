"""
app.py — India Trade Intelligence Dashboard
Run: streamlit run app.py
"""
import io
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── Page config (must be first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="India Trade Intelligence",
    page_icon="🇮🇳", layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from preprocess import (
        load_and_preprocess, safe_nunique, validate,
        HS2_LABELS, COUNTRY_NAMES, FLOW_NORM, get_available_levels,
    )
except Exception as _import_exc:
    import traceback
    st.error("**App failed to import dependencies. Full traceback:**")
    st.code(traceback.format_exc(), language="python")
    st.stop()

PROC = ROOT / "data" / "processed"

# ─── Hide default Streamlit chrome ────────────────────────────────────
st.markdown("""
<style>
#MainMenu{visibility:hidden}footer{visibility:hidden}
.block-container{padding-top:1.5rem;padding-bottom:1rem}
[data-testid="stMetricLabel"]{font-size:.8rem;color:#6B7280}
[data-testid="stMetricValue"]{font-size:1.6rem;font-weight:700}
.st-emotion-cache-1wmy9hl{border-radius:12px}
div[data-testid="stHorizontalBlock"]{gap:0.5rem}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────
GT_COLORS = {"INTERMEDIATE":"#3B82F6","CAPITAL":"#10B981",
             "FINAL":"#F59E0B","UNCLASSIFIED":"#9CA3AF"}
RISK_COLORS = {"CRITICAL":"#DC2626","HIGH":"#EA580C",
               "MEDIUM":"#D97706","LOW":"#16A34A"}
HS4_DESC = {
    "8501":"Electric motors & generators","8502":"Electric generating sets",
    "8503":"Parts for motors/generators","8504":"Electrical transformers",
    "8505":"Electromagnets","8506":"Primary cells & batteries",
    "8507":"Storage batteries","8508":"Vacuum cleaners",
    "8509":"Electromech. domestic appliances","8510":"Shavers & hair clippers",
    "8511":"Ignition equipment","8512":"Vehicle lighting",
    "8513":"Portable electric lamps","8514":"Industrial electric furnaces",
    "8515":"Welding machines","8516":"Electric water heaters",
    "8517":"Smartphones & telecom","8518":"Microphones & speakers",
    "8519":"Sound recording apparatus","8521":"Video recording apparatus",
    "8522":"Parts for sound/video","8523":"Recording media",
    "8524":"Flat panel display modules","8525":"Transmission apparatus",
    "8526":"Radar & radio navigation","8527":"Radio receivers",
    "8528":"Monitors, projectors, TVs","8529":"Parts for transmission",
    "8530":"Railway signalling","8531":"Electric signalling",
    "8532":"Electrical capacitors","8533":"Electrical resistors",
    "8534":"Printed circuits","8535":"Switching apparatus ≥80V",
    "8536":"Switching apparatus <80V","8537":"Electrical control boards",
    "8538":"Parts for switching","8539":"Filament & discharge lamps",
    "8540":"Thermionic valves & tubes","8541":"Semiconductor devices",
    "8542":"Electronic integrated circuits","8543":"Electrical machines NES",
    "8544":"Insulated wire & cable","8545":"Carbon electrodes",
    "8546":"Electrical insulators","8547":"Insulating fittings",
    "8548":"Battery waste & scrap","8549":"Electrical waste NES",
}

def to_csv(df): return df.to_csv(index=False).encode("utf-8")
def dl(df, label, fname):
    st.download_button(f"⬇ Download {label}", data=to_csv(df),
                       file_name=fname, mime="text/csv")
def fmt(v):
    v = abs(v)
    if v >= 100: return f"${v:.0f}bn"
    if v >= 10:  return f"${v:.1f}bn"
    return f"${v:.2f}bn"

# ─── Data loading ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading trade data…")
def load_fact():
    p = PROC/"fact_trade_flows.parquet"
    if not p.exists(): return pd.DataFrame()
    df = load_and_preprocess(p)
    if "hs4" in df.columns:
        df["hs4_desc"] = df["hs4"].map(HS4_DESC).fillna("HS"+df["hs4"].fillna("").astype(str))
    # COUNTRY_NAMES first; text Comtrade names kept; numeric codes → "Other (code)"
    if "partner_iso3" in df.columns:
        mapped = df["partner_iso3"].astype(str).map(COUNTRY_NAMES)
        if "partner_name" not in df.columns:
            df["partner_name"] = mapped.fillna(
                "Other (" + df["partner_iso3"].astype(str) + ")"
            )
        else:
            comtrade = df["partner_name"].astype(str).str.strip()
            is_numeric = comtrade.str.match(r"^\d+$", na=False)
            fallback = comtrade.where(
                ~is_numeric,
                "Other (" + df["partner_iso3"].astype(str) + ")",
            )
            df["partner_name"] = mapped.fillna(fallback)
    return df

@st.cache_data(show_spinner="Loading sector summary…")
def load_summary():
    p = PROC/"sector_summary.parquet"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p)
    df["hs2_label"] = df["hs2"].astype(str).map(HS2_LABELS).fillna("HS"+df["hs2"].astype(str))
    return df

@st.cache_data(show_spinner="Loading tariff profile…")
def load_tariff_profile():
    p = PROC / "tariff_profile_hs4.parquet"
    if not p.exists(): return pd.DataFrame()
    return pd.read_parquet(p)

@st.cache_data(show_spinner="Loading IDS analysis…")
def load_inv_duty():
    p = PROC / "inverted_duty_analysis.parquet"
    if not p.exists(): return pd.DataFrame()
    return pd.read_parquet(p)

@st.cache_data(show_spinner="Loading duty by type…")
def load_duty_by_type():
    p = PROC / "tariff_by_good_type.parquet"
    if not p.exists(): return pd.DataFrame()
    return pd.read_parquet(p)

@st.cache_data(show_spinner="Loading HS6 tariff data…")
def load_tariff_hs6():
    p = PROC / "tariff_hs6_agg.parquet"
    if not p.exists(): return pd.DataFrame()
    return pd.read_parquet(p)

# ─── Analytics helpers ────────────────────────────────────────────────
def annual_balance(df):
    x = df[df["flow"]=="X"].groupby("year")["value_usd"].sum()
    m = df[df["flow"]=="M"].groupby("year")["value_usd"].sum()
    r = pd.DataFrame({"exports_usd":x,"imports_usd":m}).reset_index().fillna(0)
    r["balance_usd"] = r["exports_usd"]-r["imports_usd"]
    r["exports_bn"]  = r["exports_usd"]/1e9
    r["imports_bn"]  = r["imports_usd"]/1e9
    r["balance_bn"]  = r["balance_usd"]/1e9
    r["cover"]       = r["exports_usd"]/r["imports_usd"].replace(0,float("nan"))
    return r.sort_values("year")

def top_partners(df, flow, n=10):
    r = (df[df["flow"]==flow]
         .groupby(["partner_iso3","partner_name"])["value_usd"]
         .sum().reset_index().sort_values("value_usd",ascending=False).head(n))
    r["value_bn"]  = r["value_usd"]/1e9
    r["share_pct"] = 100*r["value_usd"]/r["value_usd"].sum()
    return r.reset_index(drop=True)

def top_products(df, flow, level, n=15):
    sub = df[df["flow"]==flow].copy()
    if level not in sub.columns or sub[level].isna().all():
        for l in ["hs6","hs4","hs2"]:
            if l in sub.columns and sub[l].notna().any():
                level = l; break
    sub = sub[sub[level].notna()]
    if sub.empty: return pd.DataFrame()
    grp = [level]
    if "good_type" in sub.columns: grp.append("good_type")
    # Use hs4_desc if at hs4 level
    if level=="hs4" and "hs4_desc" in sub.columns: grp.append("hs4_desc")
    elif "product_desc" in sub.columns and sub["product_desc"].fillna("").ne("").any():
        grp.append("product_desc")
    r = (sub.groupby(grp,dropna=False)["value_usd"]
         .sum().reset_index().sort_values("value_usd",ascending=False).head(n))
    r["value_bn"]  = r["value_usd"]/1e9
    tot = df[df["flow"]==flow]["value_usd"].sum()
    r["share_pct"] = 100*r["value_usd"]/tot if tot>0 else 0
    # Build display label — always "CODE — Description" or "HS CODE"
    code_col = r[level].astype(str)
    if level == "hs4":
        desc = (r["hs4_desc"] if "hs4_desc" in r.columns
                else code_col.map(HS4_DESC))
        r["label"] = code_col + " — " + desc.fillna("HS " + code_col)
    elif level == "hs6":
        desc = code_col.map(hs6_desc) if hs6_desc else pd.Series("", index=r.index)
        if "product_desc" in r.columns:
            explicit = r["product_desc"].fillna("").astype(str).str.strip()
            desc = explicit.where(explicit != "", desc)
        r["label"] = code_col + " — " + desc.fillna("HS " + code_col)
    else:
        if "product_desc" in r.columns and r["product_desc"].fillna("").ne("").any():
            r["label"] = code_col + " — " + r["product_desc"].fillna("HS " + code_col).astype(str)
        else:
            r["label"] = code_col
    return r.reset_index(drop=True)

def china_dep(df):
    imp = df[df["flow"]=="M"].copy()
    if "hs4" not in imp.columns or imp["hs4"].isna().all():
        return pd.DataFrame()
    total = imp.groupby("hs4")["value_usd"].sum().rename("total_usd")
    china = (imp[imp["partner_iso3"].astype(str)=="156"]
             .groupby("hs4")["value_usd"].sum().rename("china_usd"))
    gtype = imp.groupby("hs4")["good_type"].first()
    r = pd.concat([total,china,gtype],axis=1).reset_index()
    r["china_usd"]   = r["china_usd"].fillna(0)
    r["china_share"] = r["china_usd"]/r["total_usd"]
    r["total_bn"]    = r["total_usd"]/1e9
    r["china_bn"]    = r["china_usd"]/1e9
    r["hs4_desc"]    = r["hs4"].map(HS4_DESC).fillna("HS"+r["hs4"].astype(str))
    def risk(row):
        s,g = row["china_share"],row.get("good_type","")
        if s>0.7 and g in("INTERMEDIATE","CAPITAL"): return "CRITICAL"
        if s>0.5 and g in("INTERMEDIATE","CAPITAL"): return "HIGH"
        if s>0.3: return "MEDIUM"
        return "LOW"
    r["risk_level"] = r.apply(risk,axis=1)
    return r.sort_values("china_share",ascending=False)

def calc_rca(df):
    exp = df[df["flow"]=="X"].copy()
    if "hs4" not in exp.columns or exp["hs4"].isna().all():
        return pd.DataFrame()
    total = exp["value_usd"].sum()
    n     = exp["hs4"].nunique()
    r = exp.groupby(["hs4","good_type"])["value_usd"].sum().reset_index()
    r["hs4_desc"]    = r["hs4"].map(HS4_DESC).fillna("HS"+r["hs4"].astype(str))
    r["share"]       = r["value_usd"]/total if total>0 else 0
    r["rca"]         = r["share"]*n
    r["value_bn"]    = r["value_usd"]/1e9
    r["has_rca"]     = r["rca"]>1
    r["share_pct"]   = 100*r["share"]
    return r.sort_values("rca",ascending=False)

# ─── Load data ────────────────────────────────────────────────────────
df_fact    = load_fact()
df_summary = load_summary()

# HS6 product description lookup — '850110' → 'Electric motors <75W'
hs6_desc: dict = {}
if not df_fact.empty and "hs6" in df_fact.columns and "product_desc" in df_fact.columns:
    hs6_desc = (
        df_fact[df_fact["product_desc"].notna() & df_fact["product_desc"].ne("")]
        .drop_duplicates("hs6")
        .set_index("hs6")["product_desc"]
        .to_dict()
    )

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇮🇳 India Trade Intelligence")
    st.caption("UN Comtrade+ Bulk | 2014–2024 | NCAER")
    st.divider()

    page = st.radio("", ["🌐 Global Overview","🔍 Sector Deep Dive"],
                    label_visibility="collapsed")

    if page=="🔍 Sector Deep Dive" and not df_fact.empty:
        st.divider()
        # Year
        yrs = sorted([int(y) for y in df_fact["year"].dropna().unique()])
        if len(yrs)>=2:
            yr = st.select_slider("📅 Years",options=yrs,value=(min(yrs),max(yrs)))
        elif len(yrs)==1:
            yr=(yrs[0],yrs[0])
            st.caption(f"Only year {yrs[0]} in data. Re-run pipeline --sector 85")
        else:
            yr=(2014,2024)
        # Flow
        flow_sel = st.radio("📦 Flow",
                            ["Both","Exports only","Imports only"],
                            horizontal=True)
        # Chapter
        hs2_raw = sorted(df_fact["hs2"].dropna().unique().tolist())
        ch_opts = {"🌍 All chapters":"All"}
        for h in hs2_raw:
            ch_opts[f"HS{h} — {HS2_LABELS.get(str(h),str(h))}"] = h
        ch_lbl = st.selectbox("📂 HS Chapter",list(ch_opts.keys()))
        ch_sel = ch_opts[ch_lbl]
        # Partner
        all_p = ["All partners"]+sorted(
            df_fact["partner_name"].dropna().unique().tolist())
        p_sel = st.selectbox("🤝 Partner",all_p)
        # Sub-tab
        st.divider()
        sub = st.radio("📊 Analysis",[
            "📊 Trade Balance","🌍 Partners","📦 Products",
            "🇨🇳 China Risk","📈 RCA","🏷️ Tariffs",
        ], label_visibility="collapsed")
    else:
        yr=(2014,2024); flow_sel="Both"
        ch_sel="All"; p_sel="All partners"; sub="📊 Trade Balance"

    st.divider()
    n = len(df_fact) if not df_fact.empty else 0
    st.caption(f"📁 {n:,} records | HS85 Electronics")
    if df_summary.empty:
        st.caption("⚠ Showing HS85 only — sector_summary.parquet not found")

def apply_filters(df):
    if df.empty: return df
    d = df.copy()
    d = d[d["year"]==yr[0]] if yr[0]==yr[1] else d[(d["year"]>=yr[0])&(d["year"]<=yr[1])]
    if flow_sel=="Exports only": d=d[d["flow"]=="X"]
    elif flow_sel=="Imports only": d=d[d["flow"]=="M"]
    if p_sel!="All partners": d=d[d["partner_name"]==p_sel]
    if ch_sel!="All" and "hs2" in d.columns: d=d[d["hs2"]==ch_sel]
    return d

# ══════════════════════════════════════════════════════════════════
# PAGE A — GLOBAL OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page=="🌐 Global Overview":
    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#0F2347 0%,#1B4A8C 100%);
            border-radius:12px;padding:36px 40px 32px;margin-bottom:8px">
  <div style="color:#93C5FD;font-size:12px;letter-spacing:2.5px;
              text-transform:uppercase;font-weight:600;margin-bottom:8px">
    NCAER Research &nbsp;·&nbsp; Policy-Grade Analytics
  </div>
  <div style="color:#FFFFFF;font-size:34px;font-weight:800;line-height:1.2">
    India Trade Intelligence
  </div>
  <div style="color:#93C5FD;font-size:17px;margin-top:6px;font-weight:500">
    HS85 Electronics &amp; Electrical Equipment
  </div>
  <div style="color:#94A3B8;font-size:13px;margin-top:14px;line-height:1.7">
    UN Comtrade+ Bulk API &nbsp;·&nbsp; 2014–2024 &nbsp;·&nbsp;
    416,000 trade records &nbsp;·&nbsp; 200+ partner countries &nbsp;·&nbsp;
    CBIC Tariff Schedule 2025-26
  </div>
</div>
""", unsafe_allow_html=True)

    src = df_summary if not df_summary.empty else df_fact
    if src.empty:
        st.error("Trade data not found. Check that `data/processed/fact_trade_flows.parquet` exists.")
        st.stop()

    total_x = src[src["flow"]=="X"]["value_usd"].sum()/1e9
    total_m = src[src["flow"]=="M"]["value_usd"].sum()/1e9
    deficit = total_m - total_x
    years   = sorted(src["year"].dropna().unique().tolist())

    top_partner = "—"
    if not df_fact.empty:
        _tp = (df_fact[df_fact["flow"]=="M"]
               .groupby("partner_name")["value_usd"].sum()
               .sort_values(ascending=False))
        if not _tp.empty:
            top_partner = str(_tp.index[0])

    # ── 4 KPI cards ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🟢 Total Exports",       f"${total_x:,.0f}bn",
              help="All HS85 electronics exports, cumulative 2014–2024")
    c2.metric("🔵 Total Imports",       f"${total_m:,.0f}bn",
              help="All HS85 electronics imports, cumulative 2014–2024")
    c3.metric("⚖️ Trade Deficit",       f"${deficit:,.0f}bn",
              delta=f"India imports ${deficit:,.0f}bn more than it exports",
              delta_color="inverse")
    c4.metric("🌏 Top Import Partner",  top_partner,
              help="Highest cumulative HS85 import source, 2014–2024")

    st.divider()

    # ── 2 charts side by side ────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Annual Trade Balance — HS85 Electronics**")
        bal = annual_balance(src)
        fig1 = go.Figure()
        fig1.add_bar(x=bal["year"], y=bal["exports_bn"], name="Exports",
                     marker_color="#10B981", opacity=0.85)
        fig1.add_bar(x=bal["year"], y=bal["imports_bn"], name="Imports",
                     marker_color="#3B82F6", opacity=0.85)
        fig1.add_scatter(x=bal["year"], y=bal["balance_bn"], name="Balance",
                         mode="lines+markers",
                         line=dict(color="#DC2626", width=2.5),
                         marker=dict(size=6, color="#DC2626"))
        fig1.add_hline(y=0, line_dash="dash", line_color="#9CA3AF", opacity=0.4)
        fig1.update_layout(barmode="group", height=300,
                           yaxis_title="USD Billion",
                           legend=dict(orientation="h", y=1.1),
                           hovermode="x unified",
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=8,b=8,l=8,r=8))
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        st.markdown("**Top 5 Import Sources (cumulative 2014–2024)**")
        if not df_fact.empty:
            _tp5 = (df_fact[df_fact["flow"]=="M"]
                    .groupby("partner_name")["value_usd"].sum()
                    .sort_values(ascending=False).head(5).reset_index())
            _tp5["value_bn"] = _tp5["value_usd"] / 1e9
            if not _tp5.empty:
                fig2 = px.bar(_tp5.sort_values("value_bn"),
                              x="value_bn", y="partner_name", orientation="h",
                              color="value_bn",
                              color_continuous_scale=[[0,"#BFDBFE"],[1,"#1D4ED8"]],
                              text="value_bn",
                              labels={"value_bn":"USD Billion","partner_name":""})
                fig2.update_traces(texttemplate="$%{text:.0f}bn", textposition="outside")
                fig2.update_layout(height=300, plot_bgcolor="white",
                                   showlegend=False, coloraxis_showscale=False,
                                   margin=dict(t=8,b=8,l=8,r=80))
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── What this dashboard covers ────────────────────────────────────────────
    st.markdown("#### What this dashboard covers")
    t1,t2,t3,t4 = st.columns(4)
    _tiles = [
        (t1, "📊", "Trade Flows",           "#3B82F6",
         "HS4 and HS6 bilateral exports &amp; imports across 11 years and 200+ partner countries."),
        (t2, "🇨🇳", "China Risk",            "#EF4444",
         "HS6-level China import dependency. Flags CRITICAL inputs sourced &gt;70% from China."),
        (t3, "📈", "Export Competitiveness","#10B981",
         "Intra-sector RCA at HS6. Reveals where India holds revealed comparative advantage."),
        (t4, "🏷️", "Tariff Structure",      "#F59E0B",
         "CBIC 2025-26 BCD at HS6 with Inverted Duty Structure analysis and China overlay."),
    ]
    for col, icon, title, color, desc in _tiles:
        with col:
            st.markdown(f"""
<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
            padding:20px 16px;border-top:3px solid {color};min-height:148px">
  <span style="font-size:26px">{icon}</span>
  <div style="font-weight:700;color:#1E293B;font-size:15px;margin:8px 0 6px">{title}</div>
  <div style="font-size:12px;color:#64748B;line-height:1.55">{desc}</div>
</div>""", unsafe_allow_html=True)

    # ── Data citation ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#F1F5F9;border-radius:8px;padding:14px 18px;
            font-size:12px;color:#64748B;line-height:1.8">
  <b>Data sources:</b>
  UN Comtrade+ Bulk API — India as reporter (code 699), 2014–2024 &nbsp;·&nbsp;
  CBIC Custom Tariff Schedule 2025-26 (Ministry of Finance, GoI) &nbsp;·&nbsp;
  BEC Rev.5 product classification (INTERMEDIATE / CAPITAL / FINAL) &nbsp;·&nbsp;
  NCAER Research
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE B — SECTOR DEEP DIVE
# ══════════════════════════════════════════════════════════════════
else:
    if df_fact.empty:
        st.error("Trade data not found. Check that `data/processed/fact_trade_flows.parquet` exists.")
        st.stop()

    df = apply_filters(df_fact)
    ch_name = (HS2_LABELS.get(str(ch_sel),f"HS{ch_sel}")
               if ch_sel!="All" else "All Chapters")
    yr_str  = str(yr[0]) if yr[0]==yr[1] else f"{yr[0]}–{yr[1]}"

    st.title(f"{ch_name}")
    st.caption(f"📅 {yr_str} | 📦 {flow_sel} | 🤝 {p_sel} | {len(df):,} records")

    if df.empty:
        st.warning("No data for this filter combination. Adjust the sidebar filters.")
        st.stop()

    # KPIs
    sx  = df[df["flow"]=="X"]["value_usd"].sum()/1e9
    sm  = df[df["flow"]=="M"]["value_usd"].sum()/1e9
    sd  = sm-sx
    mt  = df[df["flow"]=="M"]["value_usd"].sum()
    csh = (df[(df["flow"]=="M")&(df["partner_iso3"].astype(str)=="156")]
           ["value_usd"].sum()/mt if mt>0 else 0.0)
    np_ = safe_nunique(df,["hs6","hs4","hs2","product_code"])

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🟢 Exports",fmt(sx),
              help="Value India sold to the world in this sector & period")
    c2.metric("🔵 Imports",fmt(sm),
              help="Value India bought from the world in this sector & period")
    c3.metric("⚖️ Balance",fmt(sd),
              delta="Surplus — India earns more" if sd<=0 else "Deficit — India spends more",
              delta_color="normal" if sd<=0 else "inverse")
    c4.metric("🇨🇳 China share",f"{csh*100:.1f}%",
              help="Share of India's imports coming from China",
              delta="⚠ Concentrated" if csh>0.5 else "OK",
              delta_color="inverse" if csh>0.5 else "off")
    c5.metric("📦 Products (HS codes)",f"{np_:,}",
              help="Unique HS product codes in this filtered view")
    st.divider()

    # ── TRADE BALANCE ──────────────────────────────────────────────
    if sub=="📊 Trade Balance":
        with st.expander("💡 What am I looking at?",expanded=False):
            st.markdown("""
**Exports** = goods India *sold* to other countries (money comes IN).
**Imports** = goods India *bought* from other countries (money goes OUT).
**Trade balance** = Exports minus Imports.

- A **deficit** means India is spending more on imports than it earns from exports.
- The **X/M ratio** is exports ÷ imports. Ratio of 0.35 means India earns 35 cents for every dollar it spends.
- Rising **intermediate imports** (blue area chart) is actually a *good sign* — it means India is importing components to manufacture things.
- Rising **final goods imports** without matching exports = lost manufacturing opportunity.
            """)

        bal=annual_balance(df)
        fig=go.Figure()
        fig.add_bar(x=bal["year"],y=bal["exports_bn"],name="🟢 Exports",
                    marker_color="#10B981",opacity=0.9)
        fig.add_bar(x=bal["year"],y=bal["imports_bn"],name="🔵 Imports",
                    marker_color="#3B82F6",opacity=0.9)
        fig.add_scatter(x=bal["year"],y=bal["balance_bn"],name="Balance",
                        mode="lines+markers",
                        line=dict(color="#DC2626",width=3),
                        marker=dict(size=9,color="#DC2626",
                                    symbol="circle",
                                    line=dict(color="white",width=2)))
        fig.add_hline(y=0,line_dash="dash",line_color="#9CA3AF",opacity=0.5,
                      annotation_text="Break-even")
        fig.update_layout(barmode="group",height=440,
                          yaxis_title="USD Billion",
                          legend=dict(orientation="h",y=1.08),
                          hovermode="x unified",plot_bgcolor="white",
                          title=dict(text=f"{ch_name} — Annual Trade",font=dict(size=15)))
        st.plotly_chart(fig,use_container_width=True)
        dl(bal,"Balance data","trade_balance.csv")

        if "good_type" in df.columns and df["good_type"].ne("UNCLASSIFIED").any():
            st.subheader("What type of goods is India importing?")
            gt=(df[df["flow"]=="M"]
                .groupby(["year","good_type"])["value_usd"]
                .sum().reset_index())
            gt["value_bn"]=gt["value_usd"]/1e9
            fig2=px.area(gt,x="year",y="value_bn",color="good_type",
                         color_discrete_map=GT_COLORS,
                         labels={"value_bn":"USD Billion","good_type":"Type"},
                         title="Import breakdown: Inputs vs Machinery vs Final goods")
            fig2.update_layout(height=360,legend=dict(orientation="h",y=1.08),
                               plot_bgcolor="white")
            st.plotly_chart(fig2,use_container_width=True)
            dl(gt,"Good type breakdown","good_type.csv")

        st.subheader("Year-by-year numbers")
        d2 = bal[["year","exports_bn","imports_bn","balance_bn","cover"]].copy()
        d2 = d2.sort_values("year", ascending=False)
        d2["exports_bn"] = d2["exports_bn"].map("${:.1f}bn".format)
        d2["imports_bn"] = d2["imports_bn"].map("${:.1f}bn".format)
        d2["balance_bn"] = d2["balance_bn"].map(
            lambda x: f"🔴 −${abs(x):.1f}bn" if x<0 else f"🟢 +${x:.1f}bn")
        d2["cover"]      = d2["cover"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        d2.columns = ["Year","Exports","Imports","Deficit/Surplus","X/M ratio"]
        st.dataframe(d2, use_container_width=True, hide_index=True)

    # ── PARTNERS ───────────────────────────────────────────────────
    elif sub=="🌍 Partners":
        with st.expander("💡 How to read this",expanded=False):
            st.markdown("""
**Top partners** shows which countries India trades with most.

- **High concentration** in one country (e.g. China at 58%) is a strategic risk.
  If that country restricts trade, India's supply chain is exposed.
- **HHI (Herfindahl-Hirschman Index)** measures concentration.
  Below 0.15 = healthy diversity. Above 0.25 = dangerously concentrated.
- Tracking **trends over time** reveals which trade relationships are growing
  or shrinking — this can reflect geopolitical shifts, FTAs, or supply chain moves.
            """)

        fc=st.radio("Direction",["🔵 Imports","🟢 Exports"],horizontal=True)
        fcode="M" if "Import" in fc else "X"
        topn=st.slider("Partners to show",5,20,10)

        tp=top_partners(df,fcode,topn)
        if not tp.empty:
            col1,col2=st.columns([3,2])
            with col1:
                cscale="Blues" if fcode=="M" else "Greens"
                fig=px.bar(tp.sort_values("value_bn"),
                           x="value_bn",y="partner_name",orientation="h",
                           color="value_bn",color_continuous_scale=cscale,
                           text="value_bn",
                           labels={"value_bn":"USD Billion","partner_name":""},
                           title=f"Top {topn} {'import sources' if fcode=='M' else 'export markets'}")
                fig.update_traces(texttemplate="$%{text:.1f}bn",textposition="outside")
                fig.update_layout(height=max(350,topn*38),showlegend=False,
                                  coloraxis_showscale=False,
                                  margin=dict(r=90),plot_bgcolor="white")
                st.plotly_chart(fig,use_container_width=True)
            with col2:
                fig2=px.pie(tp,values="value_bn",names="partner_name",
                            hole=0.42,title="Market share")
                fig2.update_traces(textposition="outside",textinfo="label+percent")
                fig2.update_layout(height=max(350,topn*38),showlegend=False)
                st.plotly_chart(fig2,use_container_width=True)

            hhi=((tp["share_pct"]/100)**2).sum()
            col_h1,col_h2=st.columns(2)
            col_h1.metric("Concentration (HHI)",f"{hhi:.3f}",
                          delta="High risk" if hhi>0.25 else ("Moderate" if hhi>0.15 else "Diversified"),
                          delta_color="inverse" if hhi>0.25 else ("off" if hhi>0.15 else "normal"),
                          help="0=perfectly diversified, 1=one country dominates")
            col_h2.metric("Top partner share",f"{tp.iloc[0]['share_pct']:.1f}%",
                          delta=tp.iloc[0]["partner_name"])
            dl(tp,"Partner data","top_partners.csv")

            st.subheader("How has each partner's trade changed over time?")
            all_p_opts=(df[df["flow"]==fcode].groupby("partner_name")["value_usd"]
                        .sum().sort_values(ascending=False).head(12).index.tolist())
            sel_p=st.multiselect("Select partners",all_p_opts,default=all_p_opts[:5])
            if sel_p:
                tr=(df[(df["flow"]==fcode)&(df["partner_name"].isin(sel_p))]
                    .groupby(["year","partner_name"])["value_usd"]
                    .sum().reset_index())
                tr["value_bn"]=tr["value_usd"]/1e9
                fig3=px.line(tr,x="year",y="value_bn",color="partner_name",
                             markers=True,
                             labels={"value_bn":"USD Billion","partner_name":""},
                             title="Partner trends over time")
                fig3.update_layout(height=380,plot_bgcolor="white",
                                   legend=dict(orientation="h",y=1.08))
                st.plotly_chart(fig3,use_container_width=True)
                dl(tr,"Partner trend","partner_trend.csv")

    # ── PRODUCTS ───────────────────────────────────────────────────
    elif sub=="📦 Products":
        with st.expander("💡 What are HS codes?",expanded=False):
            st.markdown("""
The **Harmonised System (HS)** is the global standard for classifying trade goods —
used by every customs department in the world.

| Level | Digits | Example | Meaning |
|-------|--------|---------|---------|
| HS2  | 2 | HS85 | All electrical machinery |
| HS4  | 4 | HS8517 | Telephone & telecom equipment |
| HS6  | 6 | HS851712 | Smartphones specifically |

More digits = more specific product. Use HS6 for detailed product analysis.
HS8 is India's national tariff schedule level (available if tariff-line data was loaded).
            """)

        ca,cb,cc=st.columns(3)
        with ca:
            fc_p=st.radio("Direction",["🔵 Imports","🟢 Exports"],horizontal=True)
        fcode2="M" if "Import" in fc_p else "X"

        # Detect available levels dynamically
        avail=get_available_levels(df)
        level_display={
            "hs8":"HS8 — Tariff line (most detailed)",
            "hs6":"HS6 — Sub-heading",
            "hs4":"HS4 — Heading",
            "hs2":"HS2 — Chapter",
        }
        avail_labels=[level_display.get(l,l) for l in avail]

        with cb:
            if len(avail)>1:
                # Default to HS6 if available, else HS4
                default_idx = 0  # finest level first
                l_disp=st.radio("Detail level",avail_labels,
                                index=default_idx,horizontal=False)
                l_sel=avail[avail_labels.index(l_disp)]
            else:
                l_sel=avail[0]
                st.caption(f"Level: {level_display.get(l_sel,l_sel)}")

        with cc:
            topn2=st.slider("Products",5,30,15)

        tp2=top_products(df,fcode2,l_sel,topn2)
        if tp2.empty:
            st.warning(f"No data at {l_sel} level for current filters.")
        else:
            ycol="label" if "label" in tp2.columns else l_sel
            ccol="good_type" if "good_type" in tp2.columns else None
            fig=px.bar(tp2.sort_values("value_bn"),
                       x="value_bn",y=ycol,
                       color=ccol,color_discrete_map=GT_COLORS if ccol else None,
                       orientation="h",
                       labels={"value_bn":"USD Billion",ycol:"","good_type":"Type"},
                       text="value_bn",
                       title=f"Top {topn2} {'imported' if fcode2=='M' else 'exported'} "
                             f"products ({l_sel.upper()} level)")
            fig.update_traces(texttemplate="$%{text:.2f}bn",textposition="outside")
            fig.update_layout(height=max(420,topn2*28),
                              legend=dict(orientation="h",y=1.02),
                              margin=dict(r=90),plot_bgcolor="white")
            st.plotly_chart(fig,use_container_width=True)

            show=tp2[[c for c in [l_sel,ycol,"good_type","value_bn","share_pct"]
                       if c in tp2.columns and c!=l_sel or c==l_sel]].copy()
            show=tp2[[c for c in tp2.columns
                       if c in [l_sel,"label","good_type","value_bn","share_pct"]]].copy()
            if "value_bn" in show.columns:
                show["value_bn"]=show["value_bn"].map("${:.2f}bn".format)
            if "share_pct" in show.columns:
                show["share_pct"]=show["share_pct"].map("{:.1f}%".format)
            st.dataframe(show,use_container_width=True,hide_index=True)
            dl(tp2,"Product data","top_products.csv")

            # Partner drill-down
            if l_sel in("hs4","hs6") and l_sel in tp2.columns:
                st.divider()
                st.subheader(f"🔎 Which countries supply a specific product?")
                code_opts=tp2[l_sel].tolist()
                lbl_opts=tp2["label"].tolist() if "label" in tp2.columns else code_opts
                sel_c=st.selectbox(
                    f"Select {l_sel.upper()} product",code_opts,
                    format_func=lambda x: lbl_opts[code_opts.index(x)] if x in code_opts else x
                )
                if sel_c:
                    dr=(df[(df["flow"]==fcode2)&(df[l_sel]==sel_c)]
                        .groupby("partner_name")["value_usd"]
                        .sum().reset_index()
                        .sort_values("value_usd",ascending=False).head(12))
                    dr["value_bn"]=dr["value_usd"]/1e9
                    dr["share"]=100*dr["value_usd"]/dr["value_usd"].sum()
                    fig3=px.bar(dr,x="partner_name",y="value_bn",
                                color="value_bn",
                                color_continuous_scale="Reds" if fcode2=="M" else "Greens",
                                text="value_bn",
                                title=f"Source countries — {sel_c}")
                    fig3.update_traces(texttemplate="$%{text:.2f}bn")
                    fig3.update_layout(height=360,showlegend=False,
                                       coloraxis_showscale=False,
                                       plot_bgcolor="white")
                    st.plotly_chart(fig3,use_container_width=True)
                    dl(dr,"Country breakdown","country_drilldown.csv")

            # Trend
            if l_sel=="hs4" and "hs4" in df.columns:
                st.divider()
                st.subheader("How has demand changed over time?")
                prod_opts=(tp2["label"].tolist() if "label" in tp2.columns
                           else tp2[l_sel].tolist())
                sel_pr=st.multiselect("Select products",prod_opts,default=prod_opts[:4])
                if sel_pr:
                    # Map label back to hs4
                    if "label" in tp2.columns:
                        label_to_hs4=dict(zip(tp2["label"],tp2["hs4"]))
                        hs4_sel=[label_to_hs4[x] for x in sel_pr if x in label_to_hs4]
                    else:
                        hs4_sel=sel_pr
                    if hs4_sel:
                        tr2=(df[(df["flow"]==fcode2)&(df["hs4"].isin(hs4_sel))]
                             .groupby(["year","hs4","hs4_desc"])["value_usd"]
                             .sum().reset_index())
                        tr2["value_bn"]=tr2["value_usd"]/1e9
                        tr2["label"]=tr2["hs4"]+" – "+tr2["hs4_desc"]
                        fig4=px.line(tr2,x="year",y="value_bn",color="label",
                                     markers=True,
                                     labels={"value_bn":"USD Billion","label":"Product"})
                        fig4.update_layout(height=380,plot_bgcolor="white",
                                           legend=dict(orientation="h",y=1.08))
                        st.plotly_chart(fig4,use_container_width=True)
                        dl(tr2,"Product trend","product_trend.csv")

    # ── CHINA RISK ─────────────────────────────────────────────────
    elif sub=="🇨🇳 China Risk":
        with st.expander("💡 Why does China dependence matter?",expanded=False):
            st.markdown("""
India's supply chains are deeply intertwined with China. If China restricts exports
of critical components (as it has done with rare earths, gallium, and germanium),
Indian manufacturers face immediate shortages.

**Risk levels:**
| Level | Meaning |
|-------|---------|
| 🔴 CRITICAL | China supplies >70% of an *input* (component/machinery) |
| 🟠 HIGH | China supplies >50% of an input |
| 🟡 MEDIUM | China supplies >30% of any good |
| 🟢 LOW | India is not heavily exposed |

**Why inputs (INTERMEDIATE/CAPITAL) matter more than final goods:**
High China dependence in finished TVs can be substituted with Vietnamese TVs.
But high dependence in *display panels* that go into every TV assembled in India
means the vulnerability runs through the entire domestic industry.
            """)

        with st.expander("📐 Methodology",expanded=False):
            st.markdown("""
**China share** = China's bilateral imports ÷ total India imports, at the HS4 level.

`china_share(hs4) = Σ value_usd [partner=156, hs4] / Σ value_usd [hs4]`

**Data source:** UN Comtrade+ bulk API, India as reporter (code 699), 2014–2024.
Partner code 156 = People's Republic of China. Excludes re-exports via Hong Kong (344) or other intermediaries.

**Risk classification thresholds:**

| Level | Condition |
|---|---|
| CRITICAL | china_share > 70% AND good_type ∈ {INTERMEDIATE, CAPITAL} |
| HIGH | china_share > 50% AND good_type ∈ {INTERMEDIATE, CAPITAL} |
| MEDIUM | china_share > 30% (any good type) |
| LOW | otherwise |

**good_type** is from BEC Rev.5 classification: INTERMEDIATE = components/materials, CAPITAL = machinery/equipment, FINAL = consumer goods. Inputs matter more because substitution is harder than for finished goods.
            """)

        imp_all = df[df["flow"]=="M"].copy()
        if imp_all.empty or "hs6" not in imp_all.columns or imp_all["hs6"].isna().all():
            st.warning("China risk requires hs6 import data. Check your pipeline output.")
        else:
            tot6 = imp_all.groupby("hs6")["value_usd"].sum().rename("total_usd")
            chn6 = (imp_all[imp_all["partner_iso3"].astype(str)=="156"]
                    .groupby("hs6")["value_usd"].sum().rename("china_usd"))
            cd = pd.concat([tot6, chn6], axis=1).reset_index().fillna(0)
            cd["china_share"]  = (cd["china_usd"] /
                                   cd["total_usd"].replace(0, float("nan"))).fillna(0)
            cd["total_bn"]     = cd["total_usd"] / 1e9
            cd["china_bn"]     = cd["china_usd"] / 1e9
            cd["product_desc"] = cd["hs6"].map(hs6_desc).fillna("HS " + cd["hs6"].astype(str))
            gtype = imp_all.groupby("hs6")["good_type"].first().reset_index()
            cd = cd.merge(gtype, on="hs6", how="left")
            cd["good_type"] = cd["good_type"].fillna("UNCLASSIFIED")
            def _crisk(row):
                s, g = row["china_share"], row["good_type"]
                if s>0.7 and g in("INTERMEDIATE","CAPITAL"): return "CRITICAL"
                if s>0.5 and g in("INTERMEDIATE","CAPITAL"): return "HIGH"
                if s>0.3: return "MEDIUM"
                return "LOW"
            cd["risk_level"] = cd.apply(_crisk, axis=1)
            cd = cd[cd["total_usd"]>0].sort_values("china_share", ascending=False)

            crit      = int((cd["risk_level"]=="CRITICAL").sum())
            high      = int((cd["risk_level"]=="HIGH").sum())
            china_tot = float(cd["china_bn"].sum())
            avg       = float(cd["china_share"].mean())

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🔴 CRITICAL HS6 lines", str(crit),
                      delta="Immediate policy priority", delta_color="inverse")
            c2.metric("🟠 HIGH risk lines",     str(high),
                      delta="Monitor closely",          delta_color="inverse")
            c3.metric("Average China share",    f"{avg*100:.0f}%")
            c4.metric("Total from China",       f"${china_tot:.0f}bn")

            cd["china_pct"] = cd["china_share"] * 100
            chart_cd = cd.nlargest(30, "china_share").sort_values("china_share")
            fig = px.bar(chart_cd,
                         x="china_pct", y="product_desc",
                         color="risk_level", color_discrete_map=RISK_COLORS,
                         orientation="h",
                         labels={"china_pct":"China share (%)","product_desc":"",
                                 "risk_level":"Risk"},
                         text="china_pct",
                         hover_data={"hs6":True,"total_bn":":.1f","china_bn":":.1f"},
                         title="Top 30 HS6 Lines by China Import Dependency")
            fig.add_vline(x=50, line_dash="dash", line_color="#DC2626",
                          annotation_text="50% danger zone",
                          annotation_position="top right")
            fig.add_vline(x=70, line_dash="dot",  line_color="#7f0000",
                          annotation_text="70% critical",
                          annotation_position="bottom right")
            fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig.update_layout(height=max(500, len(chart_cd)*20),
                              legend=dict(orientation="h", y=1.01),
                              margin=dict(r=60, l=280), xaxis_range=[0,115],
                              plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            disp = cd[["hs6","product_desc","good_type","risk_level",
                        "china_share","china_bn","total_bn"]].copy()
            disp["china_share"] = (disp["china_share"]*100).map("{:.1f}%".format)
            disp["china_bn"]    = disp["china_bn"].map("${:.1f}bn".format)
            disp["total_bn"]    = disp["total_bn"].map("${:.1f}bn".format)
            disp.columns = ["HS6","Product","Type","Risk","China %","From China","Total Imports"]
            def _cr_style(v):
                return {"CRITICAL":"background-color:#FEE2E2",
                        "HIGH":    "background-color:#FED7AA",
                        "MEDIUM":  "background-color:#FEF3C7",
                        "LOW":     "background-color:#D1FAE5"}.get(v,"")
            st.dataframe(disp.style.map(_cr_style, subset=["Risk"]),
                         use_container_width=True, hide_index=True)
            dl(cd, "China risk data (HS6)", "china_risk_hs6.csv")

    # ── RCA ────────────────────────────────────────────────────────
    elif sub=="📈 RCA":
        with st.expander("💡 What is RCA?",expanded=False):
            st.markdown("""
**Revealed Comparative Advantage (Balassa, 1965)** measures whether India is
unusually specialised in exporting a given product.

**Formula (intra-sector version used here):**
> RCA = (Product's share of India's HS85 exports) × (Number of HS4 categories)

**Interpretation:**
- RCA > 1 → India exports this product more than it would if exports were evenly spread
- RCA = 19 for smartphones → India's smartphone exports are **19× more concentrated** than the average HS4 product

**Important caveat:** This is an *intra-sector* RCA (within HS85 only). A full
cross-sector Balassa RCA requires world export totals (available from WITS/Comtrade world reporter).
For policy research, supplement this with WITS cross-sector RCA data.
            """)

        with st.expander("📐 Methodology",expanded=False):
            st.markdown("""
**Balassa (1965) Revealed Comparative Advantage — intra-sector version used here:**

> RCA(i) = (share of product i in India's HS85 exports) × (number of HS4 categories)

This normalises each product's export share by what it would be if exports were evenly spread — so RCA > 1 means India is more specialised in this product than the HS85 average.

**Interpretation:**
- RCA > 1 → comparative advantage within electronics
- RCA = 1 → exactly average concentration
- RCA < 1 → below-average specialisation

**Important caveat:** This is an *intra-sector* RCA only. The standard Balassa formula uses world export totals in the denominator, which requires cross-country data from WITS or Comtrade world reporter. The values here are therefore not comparable across sectors or to published Balassa indices.
            """)

        exp_df = df[df["flow"]=="X"].copy()
        if exp_df.empty or "hs6" not in exp_df.columns or exp_df["hs6"].isna().all():
            st.warning("RCA requires export data with hs6 codes.")
        else:
            total_exp = float(exp_df["value_usd"].sum())
            n_hs6     = int(exp_df["hs6"].dropna().nunique())

            rc = (exp_df.groupby("hs6")
                  .agg(value_usd=("value_usd","sum"),
                       good_type=("good_type","first"))
                  .reset_index())
            rc["product_desc"] = rc["hs6"].map(hs6_desc).fillna("HS " + rc["hs6"].astype(str))
            rc["label"]        = rc["hs6"] + " — " + rc["product_desc"]
            rc["share"]        = rc["value_usd"] / total_exp if total_exp > 0 else 0
            rc["share_pct"]    = 100 * rc["share"]
            rc["rca"]          = rc["share"] * n_hs6
            rc["has_rca"]      = rc["rca"] > 1
            rc["value_bn"]     = rc["value_usd"] / 1e9
            rc = rc.sort_values("rca", ascending=False).reset_index(drop=True)

            nr    = int(rc["rca"].gt(1).sum())
            top_r = rc.iloc[0]
            rca_x = float(rc.loc[rc["has_rca"],"value_bn"].sum())
            tot   = float(rc["value_bn"].sum())

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("HS6 lines with RCA>1", f"{nr} / {len(rc)}",
                      help="HS6 codes where India has above-average export concentration")
            top_lbl = top_r["product_desc"]
            c2.metric("Strongest line",
                      (top_lbl[:28]+"…") if len(top_lbl)>28 else top_lbl,
                      delta=f"RCA = {top_r['rca']:.1f}×")
            c3.metric("Exports in RCA>1 lines", f"${rca_x:.1f}bn")
            c4.metric("Share of sector exports",
                      f"{100*rca_x/tot:.0f}%" if tot>0 else "—")

            fig = px.scatter(rc, x="share_pct", y="rca",
                             size="value_bn", color="good_type",
                             color_discrete_map=GT_COLORS,
                             hover_name="label",
                             hover_data={"hs6":True,"value_bn":":.2f",
                                         "rca":":.2f","share_pct":":.2f"},
                             labels={"share_pct":"Share of HS85 exports (%)",
                                     "rca":"RCA (intra-sector)",
                                     "value_bn":"Export value ($bn)"},
                             title="India's Export Competitiveness — HS85 at HS6 Level",
                             size_max=50)
            fig.add_hline(y=1, line_dash="dash", line_color="#9CA3AF",
                          annotation_text="RCA = 1 (sector average)",
                          annotation_position="right")
            fig.update_layout(height=520, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            disp = rc[["hs6","product_desc","good_type",
                        "rca","has_rca","share_pct","value_bn"]].copy()
            disp["rca"]       = disp["rca"].map("{:.2f}".format)
            disp["share_pct"] = disp["share_pct"].map("{:.2f}%".format)
            disp["value_bn"]  = disp["value_bn"].map("${:.2f}bn".format)
            disp["has_rca"]   = disp["has_rca"].map({True:"✅ Yes",False:"❌ No"})
            disp.columns = ["HS6","Product","Type","RCA Score",
                             "Competitive?","Export Share","Value"]
            st.dataframe(disp, use_container_width=True, hide_index=True)
            dl(rc, "RCA analysis (HS6)", "rca_analysis_hs6.csv")

    # ── TARIFFS ────────────────────────────────────────────────────
    elif sub=="🏷️ Tariffs":
        st.subheader("India's Tariff Policy on Electronics (CBIC 2025-26)")
        st.caption("Source: Central Board of Indirect Taxes and Customs | Official Schedule of Duties, 2025-26")

        tar_profile = load_tariff_profile()
        tar_ida     = load_inv_duty()
        tar_gtype   = load_duty_by_type()

        if tar_profile.empty:
            st.error("Run `build_tariff_parquets.py` then `tariff_analysis.py` to generate tariff analytics.")
            st.stop()

        with st.expander("💡 Tariff concepts",expanded=False):
            st.markdown("""
**BCD (Basic Customs Duty):** The primary import tariff India charges at the border.

**Total Effective Duty:** BCD + IGST + Social Welfare Surcharge — the all-in landed cost duty.

**Inverted Duty Structure (IDS):** When inputs/intermediates face *higher* BCD than the finished goods
they are used to manufacture. This penalises domestic assembly — it becomes cheaper to import the
finished product than to make it in India from imported components.

**IDS Gap (pp):** This HS6 line's BCD minus the average BCD on final goods in HS85.
Positive gap = inverted duty on that specific input line.
            """)

        # ── China exposure period selector ────────────────────────────────────
        period_mode = st.radio(
            "China exposure period",
            ["Full period (2014–2024)", "Recent 3-year avg (2022–2024)"],
            horizontal=True,
            key="tariff_period",
        )
        _yr_list = [2022, 2023, 2024] if "Recent" in period_mode else None

        _imp = df_fact[(df_fact["hs2"] == "85") & (df_fact["flow"] == "M")].copy()
        if _yr_list:
            _imp = _imp[_imp["year"].isin(_yr_list)]

        _t6 = _imp.groupby("hs6")["value_usd"].sum().rename("_t")
        _c6 = (_imp[_imp["partner_iso3"].astype(str) == "156"]
               .groupby("hs6")["value_usd"].sum().rename("_c"))
        _cs6 = pd.concat([_t6, _c6], axis=1).reset_index().fillna(0)
        _cs6["china_share"] = (_cs6["_c"] / _cs6["_t"].replace(0, float("nan"))).fillna(0)

        _t4 = _imp.groupby("hs4")["value_usd"].sum().rename("_t")
        _c4 = (_imp[_imp["partner_iso3"].astype(str) == "156"]
               .groupby("hs4")["value_usd"].sum().rename("_c"))
        _cs4 = pd.concat([_t4, _c4], axis=1).reset_index().fillna(0)
        _cs4["china_share"] = (_cs4["_c"] / _cs4["_t"].replace(0, float("nan"))).fillna(0)

        def _risk(share, lbl):
            if share > 0.7 and lbl == "INPUT": return "CRITICAL"
            if share > 0.5 and lbl == "INPUT": return "HIGH"
            if share > 0.3: return "MEDIUM"
            return "LOW"

        st.info(
            "**BCD (Basic Customs Duty)** is the base tariff rate applied at the border. "
            "**Total Effective Duty = BCD + IGST + SWS**, where IGST (Integrated GST) is typically 18% "
            "on imports and SWS (Social Welfare Surcharge) is 10% of BCD. "
            "Total Effective Duty is the full landed-cost burden on importers."
        )

        # ── Section A — Tariff Profile ─────────────────────────────────────────
        st.markdown("#### A — Tariff Structure by HS4 Group")

        tar_a = tar_profile.copy()
        tar_a["hs4_desc"] = tar_a["hs4"].map(HS4_DESC).fillna("HS" + tar_a["hs4"].astype(str))
        tar_a = tar_a.drop(columns=["china_share"]).merge(
            _cs4[["hs4", "china_share"]], on="hs4", how="left"
        )
        tar_a["china_share"] = tar_a["china_share"].fillna(0)

        fig_a = px.bar(
            tar_a.sort_values("bcd_avg", ascending=False),
            x="hs4", y="bcd_avg",
            color="good_type",
            color_discrete_map=GT_COLORS,
            hover_name="hs4_desc",
            hover_data={"import_value_bn":":.1f","total_eff_avg":":.1f","china_share":":.2f"},
            labels={"hs4":"HS4 Group","bcd_avg":"BCD Applied (%)","good_type":"Type"},
            title="Basic Customs Duty by HS4 Group (CBIC 2025-26)",
        )
        fig_a.update_layout(height=380, plot_bgcolor="white",
                            legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_a, use_container_width=True)

        # HS6-level detail table
        tar_hs6 = load_tariff_hs6()
        if not tar_hs6.empty:
            hs85_t6 = tar_hs6[tar_hs6["hs2"] == "85"].copy()
            # Join product descriptions from trade data
            hs85_t6["product_desc"] = (
                hs85_t6["hs6"].map(hs6_desc).fillna("HS " + hs85_t6["hs6"].astype(str))
            )
            if not tar_profile.empty and "good_type" in tar_profile.columns:
                hs85_t6 = hs85_t6.merge(
                    tar_profile[["hs4","good_type"]].drop_duplicates(),
                    on="hs4", how="left",
                )
            if "good_type" not in hs85_t6.columns:
                hs85_t6["good_type"] = "UNCLASSIFIED"
            else:
                hs85_t6["good_type"] = hs85_t6["good_type"].fillna("UNCLASSIFIED")
            # Column order: HS6, Description, Good Type, BCD, IGST, Total Eff Duty, N Lines
            ta6 = hs85_t6[["hs6","product_desc","good_type",
                            "bcd_avg","igst_avg","total_eff_avg","n_lines"]].copy()
            ta6["bcd_avg"]       = ta6["bcd_avg"].round(1)
            ta6["igst_avg"]      = ta6["igst_avg"].round(1)
            ta6["total_eff_avg"] = ta6["total_eff_avg"].round(1)
            ta6.columns = ["HS6","Product Description","Good Type",
                           "BCD (%)","IGST (%)","Total Effective Duty (%)","N Lines"]
            st.dataframe(ta6, use_container_width=True, hide_index=True, height=420)
            dl(hs85_t6, "Tariff detail (HS6)", "tariff_detail_hs6.csv")
        else:
            ta = tar_a[["hs4","hs4_desc","good_type","bcd_avg","total_eff_avg",
                        "import_value_bn","ids_signal","risk_level"]].copy()
            ta["bcd_avg"]         = ta["bcd_avg"].round(1)
            ta["total_eff_avg"]   = ta["total_eff_avg"].round(1)
            ta["import_value_bn"] = ta["import_value_bn"].round(1)
            ta["ids_signal"]      = ta["ids_signal"].map({True:"⚠️", False:""})
            ta.columns = ["HS4","Product","Type","BCD (%)","Total Eff (%)","Imports ($bn)","IDS","Risk"]
            st.dataframe(ta, use_container_width=True, hide_index=True)
            dl(tar_a, "Tariff profile (HS4)", "tariff_profile_hs4.csv")

        st.divider()

        # ── Section B — Inverted Duty Structure (HS6) ─────────────────────────
        st.markdown("#### B — Inverted Duty Structure (HS6 Level)")
        st.markdown(
            "Each HS6 tariff line is labelled **INPUT** (INTERMEDIATE or CAPITAL good) or "
            "**OUTPUT** (FINAL good). The IDS gap measures how much higher an INPUT line's BCD is "
            "compared to the average BCD across all OUTPUT lines in HS85. "
            "A positive gap = inverted duty on that specific product line."
        )

        with st.expander("📐 Methodology — IDS", expanded=False):
            st.markdown("""
**Inverted Duty Structure (IDS):** an INPUT good's BCD exceeds the average BCD of OUTPUT goods in the same sector.

**Definitions:**
- **INPUT** = HS6 lines whose parent HS4 heading is classified INTERMEDIATE or CAPITAL per BEC Rev.5
- **OUTPUT** = HS6 lines whose parent HS4 heading is classified FINAL per BEC Rev.5
- **Reference BCD** = unweighted mean BCD across all OUTPUT HS6 lines in HS85 (chapter-level benchmark)
- **IDS Gap (pp)** = this INPUT line's BCD − Reference BCD
- **IDS Flag** = True where IDS Gap > 0

**Limitation:** good_type is assigned at the HS4 heading level (BEC Rev.5). All HS6 sub-headings within the same HS4 share the same INPUT/OUTPUT classification; finer-grained classification would require an HS6-level BEC concordance.

**Policy implication:** IDS raises the cost of imported inputs relative to finished goods, reducing the incentive to assemble domestically. It is a structural form of anti-manufacturing bias embedded in the tariff schedule.
            """)


        if not tar_ida.empty:
            # Rebuild china_share and risk_level for selected period
            tar_ida_disp = tar_ida.drop(columns=["china_share", "risk_level"]).merge(
                _cs6[["hs6", "china_share"]], on="hs6", how="left"
            )
            tar_ida_disp["china_share"] = tar_ida_disp["china_share"].fillna(0)
            tar_ida_disp["risk_level"]  = tar_ida_disp.apply(
                lambda r: _risk(r["china_share"], r["input_output_label"]), axis=1
            )

            ids_rows = tar_ida_disp[tar_ida_disp["ids_flag"]]
            n_ids    = int(tar_ida_disp["ids_flag"].sum())
            avg_gap  = float(ids_rows["ids_gap_pp"].mean()) if n_ids > 0 else 0.0
            imp_exp  = float(ids_rows["import_value_bn"].sum())

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("HS6 lines with IDS", str(n_ids),
                       delta="Inputs taxed above final goods average",
                       delta_color="inverse")
            mc2.metric("Avg IDS Gap", f"{avg_gap:.1f} pp",
                       help="Average BCD premium that IDS-flagged inputs carry above final goods")
            mc3.metric("IDS Import Exposure", f"${imp_exp:.0f}bn",
                       help="Cumulative import value (all dataset years) of IDS-flagged HS6 lines")

            chart_rows = ids_rows.copy()
            chart_rows["label"] = chart_rows["hs6"] + "  " + chart_rows["hs4_desc"]

            fig_b = px.bar(
                chart_rows.sort_values("ids_gap_pp"),
                x="ids_gap_pp", y="label",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                orientation="h",
                labels={"ids_gap_pp":"IDS Gap (pp above output avg BCD)",
                        "label":"","risk_level":"Risk"},
                title="IDS-Flagged HS6 Input Lines — BCD Premium above Final Goods Average",
                hover_data={"bcd_avg":":.1f","import_value_bn":":.2f","china_share":":.2f"},
            )
            fig_b.add_vline(x=0, line_dash="dash", line_color="#6B7280")
            fig_b.update_layout(
                height=max(380, n_ids * 22),
                plot_bgcolor="white",
                legend=dict(orientation="h", y=1.02),
                margin=dict(l=250),
            )
            st.plotly_chart(fig_b, use_container_width=True)

            tb = tar_ida_disp.sort_values("ids_gap_pp", ascending=False).copy()
            tb["ids_flag"]        = tb["ids_flag"].map({True:"⚠️ Yes", False:"No"})
            tb["bcd_avg"]         = tb["bcd_avg"].round(1)
            tb["ids_gap_pp"]      = tb["ids_gap_pp"].round(2)
            tb["import_value_bn"] = tb["import_value_bn"].round(2)
            tb["china_share"]     = (tb["china_share"] * 100).round(1)
            tb_disp = tb[["hs6","hs4","hs4_desc","good_type","input_output_label",
                           "bcd_avg","import_value_bn","ids_gap_pp",
                           "ids_flag","china_share","risk_level"]].copy()
            tb_disp.columns = ["HS6","HS4","Product Group","Good Type","In/Out",
                                "BCD (%)","Imports ($bn)","IDS Gap (pp)","IDS","China (%)","Risk"]
            st.dataframe(tb_disp, use_container_width=True, hide_index=True)
            _period_label = "2022–2024" if _yr_list else "2014–2024"
            st.caption(
                f"**Data note — China (%):** share of India's bilateral imports {_period_label} "
                f"from partner code 156 (China), sourced from UN Comtrade+. "
                f"Excludes re-exports via third countries."
            )
            dl(tar_ida_disp, "IDS analysis (HS6)", "ids_analysis_hs6.csv")

        st.divider()

        # ── Section C — Duty Burden by Good Type ──────────────────────────────
        st.markdown("#### C — Duty Burden by Good Type")
        st.caption("Trade-value-weighted average BCD by good type. "
                   "Higher burden on intermediates vs finals indicates structural anti-manufacturing bias.")

        if not tar_gtype.empty:
            fig_c = px.bar(
                tar_gtype,
                x="good_type", y="avg_bcd",
                color="good_type",
                color_discrete_map=GT_COLORS,
                text="avg_bcd",
                labels={"good_type":"Good Type","avg_bcd":"Weighted Avg BCD (%)"},
                title="Trade-weighted Average BCD by Good Type — HS85 (CBIC 2025-26)",
            )
            fig_c.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_c.update_layout(
                height=380, plot_bgcolor="white", showlegend=False,
                yaxis_range=[0, tar_gtype["avg_bcd"].max() * 1.3],
            )
            st.plotly_chart(fig_c, use_container_width=True)

            inter_r = tar_gtype[tar_gtype["good_type"] == "INTERMEDIATE"]
            final_r = tar_gtype[tar_gtype["good_type"] == "FINAL"]
            if not inter_r.empty and not final_r.empty:
                penalty = float(inter_r["avg_bcd"].iloc[0]) - float(final_r["avg_bcd"].iloc[0])
                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.metric("INTERMEDIATE avg BCD",
                              f"{float(inter_r['avg_bcd'].iloc[0]):.1f}%")
                col_c2.metric("FINAL avg BCD",
                              f"{float(final_r['avg_bcd'].iloc[0]):.1f}%")
                col_c3.metric("Input Tax Penalty", f"{penalty:+.2f} pp",
                              delta_color="inverse" if penalty > 0 else "normal",
                              help="Positive = inputs taxed more than final goods")
                if penalty > 0:
                    st.error(f"**Input Tax Penalty: {penalty:+.1f} pp** — "
                             f"intermediates face higher average duty than finals. "
                             f"Domestic assembly is structurally disadvantaged.")
                else:
                    st.success(f"**Input Tax Penalty: {penalty:.1f} pp** — "
                               f"trade-weighted intermediates face lower average BCD than finals in HS85. "
                               f"However, specific HS6 input lines still carry inverted duty (see Section B).")
            dl(tar_gtype, "Duty by type", "duty_by_good_type.csv")

    # ── RAW DATA ───────────────────────────────────────────────────
    with st.expander("🗃️ Raw data viewer",expanded=False):
        safe=[c for c in ["year","flow","partner_name","product_code",
                          "hs2","hs4","hs6","value_usd","qty_kg",
                          "good_type","hs_revision","hs2_label"]
              if c in df.columns]
        show=df[safe].head(500).copy()
        if "value_usd" in show.columns:
            show["value_usd"]=show["value_usd"].map("${:,.0f}".format)
        st.dataframe(show,use_container_width=True,hide_index=True)
        dl(df[safe].head(10000),"Raw data (10k rows)","raw_data.csv")
