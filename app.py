"""Finance Coach AI - Streamlit demo app.

Pipeline: CSV + Goal -> Pandas engine -> JSON context -> OpenRouter LLM advisor -> UI
"""
from __future__ import annotations

import math
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import advisor
import classifier
import engine as eng
from fake_data import generate

load_dotenv()

st.set_page_config(page_title="Finance Coach AI", page_icon="💰", layout="wide")

CURRENCIES = {"VND": "₫", "USD": "$", "JPY": "¥", "EUR": "€"}
CURRENCY_UNITS = {"VND": "dong", "USD": "dollar", "JPY": "yen", "EUR": "euro"}
CATEGORY_COLORS = {
    "Dining Out": "#34d399",
    "Groceries & Food": "#fbbf24",
    "Shopping & Tech": "#f87171",
    "Transportation": "#38bdf8",
    "Utilities & Rent": "#a78bfa",
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample_transactions.csv")
DEMO_PATH = os.path.join(BASE_DIR, "demo_upload.csv")

CSS = """
<style>
.stApp { background: #0b0f17; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }
[data-testid="stSidebar"] { background: #0e1420; border-right: 1px solid rgba(255,255,255,0.06); }
h1, h2, h3 { color: #e8edf5 !important; letter-spacing: -0.01em; }
p, li, label, .stMarkdown { color: #b8c2d4; }

.hero { background: linear-gradient(135deg,#0f1a2e 0%,#13243a 55%,#0f2431 100%); border:1px solid rgba(52,211,153,0.25);
        border-radius: 18px; padding: 22px 26px; margin-bottom: 18px; position: relative; overflow: hidden; }
.hero::after { content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px;
               background: radial-gradient(circle, rgba(52,211,153,0.18), transparent 70%); }
.hero h1 { margin:0; font-size: 1.75rem; color:#fff !important; }
.hero p { margin: 6px 0 0; color:#93a3bd; font-size: 0.95rem; }

.cards { display:flex; gap:14px; margin: 4px 0 18px; flex-wrap: wrap; }
.card { flex:1 1 190px; background: rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.08);
        border-radius: 14px; padding: 14px 16px; backdrop-filter: blur(4px); }
.card-label { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; color:#7d8aa3; }
.card-value { font-size: 1.5rem; font-weight: 700; margin: 4px 0 2px; font-variant-numeric: tabular-nums; }
.card-sub { font-size: 0.76rem; color:#7d8aa3; }

.panel { background: rgba(255,255,255,0.028); border:1px solid rgba(255,255,255,0.07); border-radius: 14px;
         padding: 18px 20px; margin-bottom: 16px; }
.panel h3 { margin-top: 0; font-size: 1.05rem; }

.intervention { display:flex; align-items:center; gap:12px; padding: 10px 14px; border-radius: 10px;
                background: rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.25); margin-bottom:8px; }
.intervention .num { background:#fbbf24; color:#0b0f17; font-weight:700; width:24px; height:24px; border-radius:50%;
                     display:flex; align-items:center; justify-content:center; font-size:0.8rem; flex:none; }
.intervention .txt { flex:1; color:#e8edf5; font-size:0.94rem; }
.intervention .amt { color:#fbbf24; font-weight:700; font-variant-numeric: tabular-nums; white-space:nowrap; }

.tag { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; }
.tag.red { background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.35); }
.tag.green { background:rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.35); }
.tag.amber { background:rgba(251,191,36,0.15); color:#fbbf24; border:1px solid rgba(251,191,36,0.35); }

[data-testid="stDialog"] { background: #ffffff; border: 1px solid #d7dee8; color: #111827; }
[data-testid="stDialog"] h1, [data-testid="stDialog"] h2, [data-testid="stDialog"] h3,
[data-testid="stDialog"] p, [data-testid="stDialog"] li,
[data-testid="stDialog"] .stMarkdown, [data-testid="stDialog"] label,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] {
    color: #111827 !important;
}
[data-testid="stDialog"] h1, [data-testid="stDialog"] h2, [data-testid="stDialog"] h3 {
    font-weight: 700;
}
[data-testid="stDialog"] [data-testid="stCaptionContainer"] p,
[data-testid="stDialog"] small {
    color: #475569 !important;
    font-weight: 500;
}
[data-testid="stDialog"] [data-testid="stDialogCloseButton"] button,
[data-testid="stDialog"] [aria-label="Close"] {
    color: #1f2937 !important;
}
[data-testid="stDialog"] .stButton > button {
    color: #111827 !important;
    background: #ffffff;
    border: 1px solid #cbd5e1;
}
[data-testid="stDialog"] .stButton > button[kind="primary"],
[data-testid="stDialog"] .stButton > button[data-testid="baseButton-primary"] {
    color: #ffffff !important;
    background: #ff4b4b;
    border-color: #ff4b4b;
}

.stButton>button, .stDownloadButton>button { border-radius: 10px; font-weight:600; }
.stSegmentedControl { margin: 2px 0 10px; }
[data-testid="stSegmentedControl"] { border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 18px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def fmt(v: float | None, cur: str, nd: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isinf(v)):
        return "—"
    sym = CURRENCIES[cur]
    if cur == "VND":
        return f"{v:,.{nd}f} {sym}"
    return f"{sym}{v:,.{nd}f}"


def metric_card(label: str, value: str, sub: str, color: str = "#e8edf5") -> str:
    return (
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value" style="color:{color}">{value}</div>'
        f'<div class="card-sub">{sub}</div></div>'
    )


def trajectory_chart(report: dict, cur: str) -> go.Figure:
    kpi = report["kpis"]
    cum = report["cum_savings"]
    goal = report["goal"]
    cone = eng.forecast_cone(report)

    fig = go.Figure()
    x_actual = [p.to_timestamp() for p in cum.index]
    y_actual = cum.values

    # forecast cone
    fig.add_trace(go.Scatter(
        x=list(cone["dates"]) + list(cone["dates"])[::-1],
        y=list(cone["upper"]) + list(cone["lower"])[::-1],
        fill="toself", fillcolor="rgba(56,189,248,0.12)", line=dict(width=0),
        hoverinfo="skip", name="Forecast band",
    ))
    fig.add_trace(go.Scatter(
        x=cone["dates"], y=cone["mean"], mode="lines",
        line=dict(color="#38bdf8", width=2, dash="dot"),
        name="Forecast (avg pace)",
    ))

    # target line
    start = report["start_date"]
    end_t = start + pd.DateOffset(months=goal["months"])
    fig.add_trace(go.Scatter(
        x=[start, end_t], y=[0, goal["amount"]], mode="lines",
        line=dict(color="#94a3b8", width=2, dash="dash"),
        name=f"Target ({goal['months']} months)",
    ))

    # actual cumulative savings
    fig.add_trace(go.Scatter(
        x=x_actual, y=y_actual, mode="lines+markers",
        line=dict(color="#34d399", width=3), marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        name="Actual savings",
    ))

    fig.add_hline(y=goal["amount"], line=dict(color="#fbbf24", width=1, dash="dot"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=430, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#dbe4f0", size=11)),
        title=dict(text="Target vs. Actual trajectory with forecast cone", font=dict(size=15, color="#e8edf5")),
        font=dict(color="#cbd5e1"),
        hoverlabel=dict(bgcolor="#172033", bordercolor="#475569", font=dict(color="#f8fafc", size=12)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.10)", tickfont=dict(color="#cbd5e1"), title_font=dict(color="#e2e8f0"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.10)", tickfont=dict(color="#cbd5e1"), title_font=dict(color="#e2e8f0"))
    return fig


def category_bars(report: dict) -> go.Figure:
    cat_monthly = report["cat_monthly"]
    fig = go.Figure()
    for cat in cat_monthly.columns:
        fig.add_trace(go.Bar(
            x=[str(p) for p in cat_monthly.index],
            y=cat_monthly[cat].values,
            name=str(cat),
            marker_color=CATEGORY_COLORS.get(str(cat), "#94a3b8"),
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack", height=360, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(color="#dbe4f0", size=11)),
        title=dict(text="Monthly spend by category", font=dict(size=15, color="#e8edf5")),
        font=dict(color="#cbd5e1"),
        hoverlabel=dict(bgcolor="#172033", bordercolor="#475569", font=dict(color="#f8fafc", size=12)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.10)", tickfont=dict(color="#cbd5e1"), title_font=dict(color="#e2e8f0"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.10)", tickfont=dict(color="#cbd5e1"), title_font=dict(color="#e2e8f0"))
    return fig


def render_advice(res: dict, cur: str) -> None:
    if not res.get("ok"):
        st.error(f"⚠️ {res.get('error', 'Unknown error')}")
        if res.get("raw"):
            with st.expander("Raw LLM output"):
                st.code(res["raw"])
        return

    if not res.get("structured") or not res.get("data"):
        raw_text = res.get("raw") or ""
        st.markdown(raw_text)
        return


    data = res["data"]
    st.markdown('<div class="panel"><h3>🩺 Diagnosis</h3>' + f"<p>{data.get('diagnosis','')}</p></div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><h3>✂️ Mandatory interventions</h3>', unsafe_allow_html=True)
    for i, iv in enumerate(data.get("interventions", []), 1):
        amt = iv.get("expected_monthly_saving")
        amt_html = f'<span class="amt">−{fmt(amt, cur)}/mo</span>' if amt else ""
        cat = iv.get("category") or ""
        tag = f'<span class="tag amber">{cat}</span>' if cat else ""
        st.markdown(
            f'<div class="intervention"><div class="num">{i}</div><div class="txt">{iv.get("action","")} {tag}</div>{amt_html}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="panel"><h3>⏱️ Timeline critique</h3>'
        + f"<p>{data.get('timeline_critique','')}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel" style="border-color:rgba(52,211,153,0.3)"><h3 style="color:#34d399">🔥 Motivation</h3>'
        + f"<p>{data.get('motivation','')}</p></div>",
        unsafe_allow_html=True,
    )


@st.dialog("🏷️ Classify transactions?")
def suggest_classify(count: int) -> None:
    st.write(
        f"**{count} transactions** have no category yet. "
        "Classifying them enables deeper analysis — category charts, anomaly detection, and sharper AI advice."
    )
    st.caption("Classification runs automatically: Jev → LLM → keyword.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Go to Transactions", type="primary", width="stretch"):
            st.session_state["active_tab"] = "🧾 Transactions"
            st.rerun()
    with c2:
        if st.button("Not now", width="stretch"):
            st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    src = st.radio("Data source", ["📁 Sample data (fake)", "⬆️ Upload CSV"], horizontal=True)

    uploaded = None
    if src.startswith("⬆️"):
        uploaded = st.file_uploader("CSV: date, item, category?, price, income", type=["csv"])
        st.caption("Required columns: `date`, `price`, `income`. Optional: `item`, `category`.")

    goal_amount = st.number_input("🎯 Savings goal", min_value=0.0, value=50_000_000.0, step=1_000_000.0)
    goal_months = st.number_input("⏳ Within how many months?", min_value=1, value=12, step=1)
    currency = st.selectbox("💱 Currency", list(CURRENCIES.keys()), index=0)

    income_override = st.number_input(
        "Monthly income (0 = auto from CSV)",
        min_value=0.0, value=0.0, step=500_000.0,
    )
    anomaly_th = st.slider("🚨 Spend anomaly threshold (%)", 10, 100, 30, 5)

    st.divider()
    gen = st.button("🎲 Regenerate sample data")
    if gen:
        rows = generate()
        pd.DataFrame(rows).to_csv(SAMPLE_PATH, index=False)
        pd.DataFrame(rows).to_csv(DEMO_PATH, index=False)
        st.success("Sample CSV regenerated!")

# ---------------------------------------------------------------- load data
df_raw: pd.DataFrame
if src.startswith("⬆️") and uploaded is not None:
    df_raw = pd.read_csv(uploaded)
elif os.path.exists(SAMPLE_PATH):
    df_raw = pd.read_csv(SAMPLE_PATH)
else:
    df_raw = pd.DataFrame(generate())

# detect a new upload / data source so the classify dialog only pops once per file
upload_token = uploaded.name if (src.startswith("⬆️") and uploaded is not None) else "sample"
if st.session_state.get("upload_token") != upload_token:
    st.session_state["upload_token"] = upload_token
    st.session_state["prompted_classify"] = False

st.session_state.setdefault("categories_map", {})

try:
    df = eng.normalize(df_raw)
except Exception as exc:
    st.error(f"❌ Error reading data: {exc}")
    st.stop()

# apply manually classified categories
cmap = st.session_state["categories_map"]
if cmap:
    mask = df["item"].isin(cmap)
    df.loc[mask, "category"] = df.loc[mask, "item"].map(cmap)

missing_cats = df[(df["income"] == 0) & ((df["category"] == "") | (df["category"].isin(["nan", "None"])))]
uncategorized = sorted({it for it in missing_cats["item"] if it})

if uncategorized and not st.session_state.get("prompted_classify"):
    st.session_state["prompted_classify"] = True
    suggest_classify(len(uncategorized))

report = eng.compute_report(
    df, goal_amount, goal_months,
    monthly_income_override=income_override or None,
    anomaly_threshold=1 + anomaly_th / 100,
)
kpi = report["kpis"]

if kpi["avg_income"] <= 0:
    st.warning("⚠️ No income data found — savings cannot be computed. Add an `income` column (salary rows with `income > 0`) or set **Monthly income** in the sidebar.")

# ---------------------------------------------------------------- hero
st.markdown(
    f"""
    <div class="hero">
      <h1>💰 Finance Coach AI</h1>
      <p>Personal finance assistant — analyzes your spending, catches impulsive overspending, and keeps you on track toward your goal of <b>{fmt(goal_amount, currency)}</b> in <b>{int(goal_months)} months</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- metrics banner
deficit_val = fmt(kpi["monthly_deficit"], currency) if kpi["monthly_deficit"] > 0 else "0"
deficit_color = "#f87171" if kpi["monthly_deficit"] > 0 else "#34d399"

if math.isfinite(kpi["projected_months"]):
    runway_val = f"{kpi['projected_months']:.1f} months"
    runway_color = "#fbbf24" if kpi["projected_months"] > goal_months else "#34d399"
else:
    runway_val = "∞"
    runway_color = "#f87171"

proj_date = kpi["projected_date"].strftime("%d/%m/%Y") if kpi["projected_date"] is not None else "—"
rate_color = "#34d399" if kpi["net_savings_rate"] >= 0 else "#f87171"

cards = "".join([
    metric_card("Deficit / month", deficit_val, "spending exceeds income", deficit_color),
    metric_card("Runway (goal reached)", runway_val, "at current pace", runway_color),
    metric_card("Expected completion date", proj_date, f"{fmt(kpi['remaining'], currency)} left", "#e8edf5"),
    metric_card("Net savings rate", f"{kpi['net_savings_rate']*100:.1f}%", f"avg savings {fmt(kpi['avg_savings'], currency)}/mo", rate_color),
])
st.markdown(f'<div class="cards">{cards}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- navigation
TAB_OVERVIEW, TAB_ADVISOR, TAB_TX = "📊 Overview", "🤖 AI Advisor", "🧾 Transactions"
st.session_state.setdefault("active_tab", TAB_OVERVIEW)
nav = st.segmented_control(
    "Navigate",
    [TAB_OVERVIEW, TAB_ADVISOR, TAB_TX],
    key="active_tab",
    label_visibility="collapsed",
)

if nav == TAB_OVERVIEW:
    st.plotly_chart(trajectory_chart(report, currency), width="stretch")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(category_bars(report), width="stretch")
    with c2:
        st.markdown('<div class="panel"><h3>🚨 Spending anomalies (latest month)</h3>', unsafe_allow_html=True)
        if report["anomalies"]:
            for a in report["anomalies"][:5]:
                st.markdown(
                    f'<div class="intervention"><div class="txt"><b>{a["category"]}</b><br/>'
                    f'<span style="color:#7d8aa3;font-size:0.84rem">Last month ({a["latest_month"]}): {fmt(a["latest"], currency)} · '
                    f'avg of previous months: {fmt(a["previous_avg"], currency)}/mo</span></div>'
                    f'<span class="tag red">+{a["change_pct"]:.0f}%</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<p style="color:#34d399">✅ No anomalies above the threshold detected.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🧮 Deterministic calculation (transparent)"):
        st.markdown(
            f"""
            - **Avg savings/month** = Avg income − Avg expense = {fmt(kpi['avg_income'], currency)} − {fmt(kpi['avg_expense'], currency)} = **{fmt(kpi['avg_savings'], currency)}**
            - **Remaining** = Goal − Saved so far = {fmt(goal_amount, currency)} − {fmt(kpi['total_saved'], currency)} = **{fmt(kpi['remaining'], currency)}**
            - **Projected months** = Remaining ÷ Avg savings = {fmt(kpi['remaining'], currency)} ÷ {fmt(kpi['avg_savings'], currency)} = **{runway_val}**
            - **Anomaly threshold**: category spend rises > {anomaly_th}% vs. the average of previous months
            """
        )

elif nav == TAB_ADVISOR:
    st.markdown(
        '<div class="panel"><p style="margin:0">The coach diagnoses your finances, prescribes concrete cuts, and honestly '
        "critiques whether your goal is achievable. Your data is condensed into a JSON context and sent through OpenRouter "
        "(model <code>openrouter/free</code>).</p></div>",
        unsafe_allow_html=True,
    )
    ctx_json = eng.build_json_context(report, currency)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        run_btn = st.button("🚀 Run AI Coach", type="primary", width="stretch")
    with col_b:
        st.caption(f"Context JSON: {len(ctx_json):,} chars sent to the LLM")

    if run_btn:
        with st.spinner(f"Coach is scrutinizing every {CURRENCY_UNITS.get(currency, 'dollar')}... 💸"):
            res = advisor.get_advice(ctx_json)
        st.session_state["advice"] = res

    if "advice" in st.session_state:
        render_advice(st.session_state["advice"], currency)

    with st.expander("👀 View JSON context sent to the LLM"):
        st.code(ctx_json, language="json")

else:  # Transactions
    st.markdown('<div class="panel"><h3>Normalized transactions</h3>', unsafe_allow_html=True)
    show = df.copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, width="stretch", height=380)
    st.markdown("</div>", unsafe_allow_html=True)

    if uncategorized:
        st.warning(f"There are {len(uncategorized)} items without a category.")
        if st.button("🏷️ Auto-classify (Jev → LLM → keyword)"):
            with st.spinner("Classifying..."):
                mapping, method = classifier.classify_items(uncategorized)
            st.session_state["categories_map"].update(mapping)
            st.session_state["last_method"] = method
            st.rerun()
    else:
        st.success("All transactions have a category ✅")

    if st.session_state.get("last_method"):
        m = st.session_state["last_method"]
        tag = {"jev": "green", "llm": "amber", "jev+llm": "amber"}.get(m, "amber")
        st.markdown(f'<span class="tag {tag}">Classified with: {m}</span>', unsafe_allow_html=True)

    if not os.getenv("OPENROUTER_API_KEY"):
        st.info("💡 Create a `.env` file with `OPENROUTER_API_KEY=...` to enable AI classification (Jev + LLM) and the AI Coach.")
