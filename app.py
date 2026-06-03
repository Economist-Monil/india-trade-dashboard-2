"""
app.py — India Trade Intelligence Dashboard
Run: streamlit run D:\Trade_Dashboard\app.py
"""
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import sys

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from preprocess import (
    load_and_preprocess, safe_nunique, validate,
    HS2_LABELS, COUNTRY_NAMES, FLOW_NORM, get_available_levels,
)

PROC = ROOT / "data" / "processed"
# ─── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Trade Intelligence",
    page_icon="🇮🇳", layout="wide",
    initial_sidebar_state="expanded",
)

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
    # Build display label
    if level=="hs4" and "hs4_desc" in r.columns:
        r["label"] = r["hs4"].astype(str)+" – "+r["hs4_desc"].astype(str)
    elif "product_desc" in r.columns and r["product_desc"].fillna("").ne("").any():
        r["label"] = r[level].astype(str)+" – "+r["product_desc"].astype(str).str[:50]
    else:
        r["label"] = r[level].astype(str)+" – "+r[level].map(HS4_DESC).fillna("").astype(str)
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
        st.caption("⚠ Run build_sector_summary.py for all sectors")

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
    st.title("India's Place in Global Trade")
    st.markdown("> Explore India's full trade story — what we buy, what we sell, "
                "from whom, and what it means for the economy.")
    st.divider()

    src = df_summary if not df_summary.empty else df_fact
    is_full = not df_summary.empty

    if src.empty:
        st.error("No data. Run: `python run_pipeline.py --sector 85`")
        st.stop()

    total_x = src[src["flow"]=="X"]["value_usd"].sum()/1e9
    total_m = src[src["flow"]=="M"]["value_usd"].sum()/1e9
    deficit = total_m-total_x
    years   = sorted(src["year"].dropna().unique().tolist())

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🟢 Total Exports",f"${total_x:,.0f}bn",
              help="All goods India sold abroad (2014–2024)")
    c2.metric("🔵 Total Imports",f"${total_m:,.0f}bn",
              help="All goods India bought from abroad (2014–2024)")
    c3.metric("⚖️ Trade Deficit",f"${deficit:,.0f}bn",
              delta=f"India spends ${deficit:,.0f}bn more",delta_color="inverse")
    c4.metric("📅 Years covered",f"{len(years)}",
              delta=f"{min(years)} – {max(years)}" if years else "—")
    c5.metric("📦 Sectors",f"{src['hs2'].nunique()}")

    if not is_full:
        st.info("📌 Showing HS85 (Electronics) only. "
                "Run `build_sector_summary.py` for all 97 sectors.")

    st.divider()
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("📅 India's trade balance over time")
        st.caption("Green = exports (money in). Blue = imports (money out). Red line = net.")
        bal = annual_balance(src)
        fig = go.Figure()
        fig.add_bar(x=bal["year"],y=bal["exports_bn"],name="Exports",
                    marker_color="#10B981",opacity=0.85)
        fig.add_bar(x=bal["year"],y=bal["imports_bn"],name="Imports",
                    marker_color="#3B82F6",opacity=0.85)
        fig.add_scatter(x=bal["year"],y=bal["balance_bn"],name="Balance",
                        mode="lines+markers",
                        line=dict(color="#DC2626",width=2.5),
                        marker=dict(size=7,color="#DC2626"))
        fig.add_hline(y=0,line_dash="dash",line_color="#9CA3AF",opacity=0.5)
        fig.update_layout(barmode="group",height=370,
                          yaxis_title="USD Billion",
                          legend=dict(orientation="h",y=1.1),
                          hovermode="x unified",plot_bgcolor="white",
                          paper_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
        dl(bal,"Annual Balance","annual_balance.csv")

    with col2:
        st.subheader("🏭 Import share by sector")
        st.caption("Bigger box = India imports more from this category.")
        sec_m=(src[src["flow"]=="M"]
               .groupby(["hs2","hs2_label"])["value_usd"]
               .sum().reset_index())
        sec_m["value_bn"]=sec_m["value_usd"]/1e9
        sec_m=sec_m[sec_m["value_bn"]>0.5]
        fig3=px.treemap(sec_m,path=["hs2_label"],values="value_bn",
                        color="value_bn",color_continuous_scale="Blues",
                        labels={"value_bn":"Imports ($bn)"})
        fig3.update_layout(height=370,margin=dict(t=5,b=5,l=5,r=5))
        fig3.update_traces(texttemplate="%{label}<br><b>$%{value:.0f}bn</b>",
                           textfont=dict(size=11))
        st.plotly_chart(fig3,use_container_width=True)

    # Sector table
    st.subheader("📋 All sectors — exports, imports and balance")
    sec_x=(src[src["flow"]=="X"].groupby("hs2")["value_usd"].sum().rename("x"))
    sec_m2=(src[src["flow"]=="M"].groupby("hs2")["value_usd"].sum().rename("m"))
    tbl=pd.concat([sec_x,sec_m2],axis=1).reset_index().fillna(0)
    tbl["Sector"]  = "HS"+tbl["hs2"].astype(str)+" — "+tbl["hs2"].astype(str).map(HS2_LABELS).fillna("")
    tbl["Exports"] = (tbl["x"]/1e9).map("${:.1f}bn".format)
    tbl["Imports"] = (tbl["m"]/1e9).map("${:.1f}bn".format)
    b=(tbl["x"]-tbl["m"])/1e9
    tbl["Balance"] = b.map(lambda v: f"▼ ${abs(v):.1f}bn" if v<0 else f"▲ ${v:.1f}bn")
    tbl["Result"]  = b.map(lambda v: "🔴 Deficit" if v<0 else "🟢 Surplus")
    tbl["Import share"]=((tbl["m"]/tbl["m"].sum()*100).map("{:.1f}%".format))
    tbl=tbl.sort_values("m",ascending=False)
    st.dataframe(tbl[["Sector","Exports","Imports","Balance","Result","Import share"]],
                 use_container_width=True,hide_index=True,height=460)
    dl(tbl,"Sector Summary","sector_summary_export.csv")

    st.info("👉 Use **Sector Deep Dive** in the sidebar to analyse any chapter in detail — "
            "partners, products up to HS6, China risk, RCA, and tariff structure.")

# ══════════════════════════════════════════════════════════════════
# PAGE B — SECTOR DEEP DIVE
# ══════════════════════════════════════════════════════════════════
else:
    if df_fact.empty:
        st.error("No data. Run: `python run_pipeline.py --sector 85`")
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
            lambda x: f"🔴 −${abs(x):.1f}bn" if x>0 else f"🟢 +${abs(x):.1f}bn")
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

        cd=china_dep(df)
        if cd.empty:
            st.warning("China risk requires hs4 data. Check your pipeline output.")
        else:
            crit=(cd["risk_level"]=="CRITICAL").sum()
            high=(cd["risk_level"]=="HIGH").sum()
            china_tot=df[(df["flow"]=="M")&(df["partner_iso3"].astype(str)=="156")
                        ]["value_usd"].sum()/1e9
            avg=cd["china_share"].mean()

            c1,c2,c3,c4=st.columns(4)
            c1.metric("🔴 CRITICAL products",str(crit),
                      delta="Immediate policy priority",delta_color="inverse")
            c2.metric("🟠 HIGH risk products",str(high),
                      delta="Monitor closely",delta_color="inverse")
            c3.metric("Average China share",f"{avg*100:.0f}%")
            c4.metric("Total from China",f"${china_tot:.0f}bn")

            cd["china_pct"]=cd["china_share"]*100
            fig=px.bar(cd.sort_values("china_pct",ascending=True),
                       x="china_pct",y="hs4_desc",
                       color="risk_level",color_discrete_map=RISK_COLORS,
                       orientation="h",
                       labels={"china_pct":"China's share (%)","hs4_desc":"",
                               "risk_level":"Risk Level"},
                       text="china_pct",
                       hover_data={"total_bn":":.1f","china_bn":":.1f"},
                       title="China's share of India's imports by product")
            fig.add_vline(x=50,line_dash="dash",line_color="#DC2626",
                          annotation_text="50% danger zone",
                          annotation_position="top right")
            fig.add_vline(x=70,line_dash="dot",line_color="#7f0000",
                          annotation_text="70% critical",
                          annotation_position="bottom right")
            fig.update_traces(texttemplate="%{text:.0f}%",textposition="outside")
            fig.update_layout(height=max(500,len(cd)*22),
                              legend=dict(orientation="h",y=1.01),
                              margin=dict(r=60),xaxis_range=[0,108],
                              plot_bgcolor="white")
            st.plotly_chart(fig,use_container_width=True)

            disp=cd[["hs4","hs4_desc","good_type","risk_level",
                      "china_share","china_bn","total_bn"]].copy()
            disp["china_share"]=(disp["china_share"]*100).map("{:.1f}%".format)
            disp["china_bn"]=disp["china_bn"].map("${:.1f}bn".format)
            disp["total_bn"]=disp["total_bn"].map("${:.1f}bn".format)
            disp.columns=["HS4","Product","Type","Risk","China %","From China","Total Imports"]
            def cr(v):
                return {"CRITICAL":"background-color:#FEE2E2",
                        "HIGH":"background-color:#FED7AA",
                        "MEDIUM":"background-color:#FEF3C7",
                        "LOW":"background-color:#D1FAE5"}.get(v,"")
            st.dataframe(disp.style.map(cr,subset=["Risk"]),
                         use_container_width=True,hide_index=True)
            dl(cd,"China risk data","china_risk.csv")

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

        rc=calc_rca(df)
        if rc.empty:
            st.warning("RCA requires export data with hs4 codes.")
        else:
            nr=(rc["rca"]>1).sum()
            top_r=rc.iloc[0]
            rca_x=rc[rc["rca"]>1]["value_bn"].sum()
            tot=rc["value_bn"].sum()

            c1,c2,c3,c4=st.columns(4)
            c1.metric("Products with RCA>1",f"{nr} / {len(rc)}",
                      help="Sectors where India is competitively strong")
            c2.metric("Strongest competitive product",
                      top_r["hs4_desc"][:26]+"…" if len(top_r["hs4_desc"])>26 else top_r["hs4_desc"],
                      delta=f"RCA = {top_r['rca']:.1f}x specialisation")
            c3.metric("Exports in RCA>1 products",f"${rca_x:.0f}bn")
            c4.metric("Share of sector exports",
                      f"{100*rca_x/tot:.0f}%" if tot>0 else "—")

            fig=px.scatter(rc,x="share_pct",y="rca",size="value_bn",color="good_type",
                           color_discrete_map=GT_COLORS,hover_name="hs4_desc",
                           hover_data={"value_bn":":.1f","rca":":.2f","share_pct":":.1f"},
                           labels={"share_pct":"Share of sector exports (%)",
                                   "rca":"Competitiveness (RCA)",
                                   "value_bn":"Export value ($bn)"},
                           title="India's Export Competitiveness — HS85 Products",
                           size_max=60)
            fig.add_hline(y=1,line_dash="dash",line_color="#9CA3AF",
                          annotation_text="RCA=1 (average)",
                          annotation_position="right")
            fig.add_annotation(x=1,y=rc["rca"].max()*0.88,
                               text="✅ Strong & specialised",
                               showarrow=False,font=dict(color="#10B981",size=11))
            fig.add_annotation(x=rc["share_pct"].max()*0.65,y=0.4,
                               text="⚠ Large volume, low specialisation",
                               showarrow=False,font=dict(color="#F59E0B",size=11))
            fig.update_layout(height=500,plot_bgcolor="white")
            st.plotly_chart(fig,use_container_width=True)

            disp=rc[["hs4","hs4_desc","good_type","rca","has_rca",
                      "share_pct","value_bn"]].copy()
            disp["rca"]=disp["rca"].map("{:.2f}".format)
            disp["share_pct"]=disp["share_pct"].map("{:.1f}%".format)
            disp["value_bn"]=disp["value_bn"].map("${:.1f}bn".format)
            disp["has_rca"]=disp["has_rca"].map({True:"✅ Yes",False:"❌ No"})
            disp.columns=["HS4","Product","Type","RCA Score","Competitive?",
                          "Export Share","Value"]
            st.dataframe(disp,use_container_width=True,hide_index=True)
            dl(rc,"RCA analysis","rca_analysis.csv")

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
