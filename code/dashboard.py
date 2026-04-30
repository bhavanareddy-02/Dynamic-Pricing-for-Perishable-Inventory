"""
Streamlit data-storytelling dashboard for the Dynamic Pricing project.

Single scrolling page styled like a walk through a grocery store. The
sidebar lists "aisles" — clicking one jumps to that anchor on the same
page. No multi-page mode; the narrative is meant to read top-to-bottom.

Run locally:
    cd code/
    streamlit run dashboard.py

For Streamlit Community Cloud, point the app at code/dashboard.py.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fresh Pricing in Missouri",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Grocery palette
PRODUCE  = "#2D5016"   # deep leafy green
LEAF     = "#5B8C3F"   # softer green
BAG      = "#8B4513"   # paper bag brown
APPLE    = "#C44E52"   # apple red
PEACH    = "#E8956F"   # warm peach
CREAM    = "#FAF7F2"   # cream paper
PAPER    = "#F5F0E8"   # parchment
INK      = "#2C2C2C"   # ink

STRAT_PALETTE = {"No Discount": "#5B8AA6", "Fixed 20%": "#E07A5F", "Dynamic": LEAF}
PLOTLY_TEMPLATE = "simple_white"

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — grocery theme + consistent typography
# ─────────────────────────────────────────────────────────────────────────────
# Font stack:
#   Headings  → "Source Serif Pro" → Georgia → Cambria (system fallbacks)
#   Body / UI → "Inter" → system-ui → -apple-system → Helvetica (clean sans-serif)
# We use Google Fonts via @import. If the user is offline, the system fallbacks
# kick in automatically.

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+Pro:wght@600;700&display=swap');

    /* ---------- Global colour + typography ---------- */
    /* IMPORTANT: do NOT use [class*="css"] here — it stomps Streamlit's icon
       fonts and Material Symbols start rendering as raw text like
       "keyboard_double_arrow_right". Scope the font to specific elements only. */
    html, body, .stApp {{
        font-family: 'Inter', system-ui, -apple-system, 'Helvetica Neue', sans-serif;
        color: {INK};
        background-color: {CREAM} !important;
    }}
    .stApp p, .stApp li, .stApp label, .stApp .stMarkdown {{
        font-family: 'Inter', system-ui, sans-serif;
        color: {INK};
    }}
    /* Belt-and-braces: any Material icon classes that DO get caught by a
       descendant selector — force them back to their own font family. */
    .material-icons, .material-icons-outlined, .material-symbols-outlined,
    [class*="material-icons"], [class*="MaterialIcons"], [class*="material-symbols"] {{
        font-family: 'Material Icons', 'Material Symbols Outlined', sans-serif !important;
    }}

    /* ---------- Headings ---------- */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Source Serif Pro', Georgia, Cambria, serif !important;
        color: {PRODUCE} !important;
        letter-spacing: -0.015em;
        font-weight: 700 !important;
    }}
    h1 {{ font-size: 2.5rem !important; line-height: 1.15 !important; margin-top: 0.5rem !important; }}
    h2 {{ font-size: 1.85rem !important; line-height: 1.2 !important; padding-top: 0.5rem !important; }}
    h3 {{ font-size: 1.35rem !important; line-height: 1.3 !important; }}
    h4 {{ font-size: 1.1rem !important; }}

    /* ---------- Section divider ---------- */
    .aisle-divider {{
        text-align: center;
        margin: 3rem 0 2rem 0;
        color: {BAG};
        font-size: 1.3rem;
        letter-spacing: 0.4em;
    }}
    .aisle-divider::before, .aisle-divider::after {{
        content: '';
        display: inline-block;
        width: 25%;
        height: 1px;
        background: {BAG};
        opacity: 0.3;
        vertical-align: middle;
        margin: 0 1rem;
    }}

    /* ---------- Hero ---------- */
    .hero {{
        background: linear-gradient(135deg, {PAPER} 0%, {CREAM} 100%);
        border-left: 6px solid {PRODUCE};
        padding: 2.4rem 2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }}
    .hero-title {{
        font-family: 'Source Serif Pro', Georgia, serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: {PRODUCE};
        margin: 0 0 0.5rem 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }}
    .hero-sub {{
        color: {BAG};
        font-size: 1.05rem;
        font-style: italic;
        margin: 0 0 1rem 0;
        font-family: 'Source Serif Pro', Georgia, serif;
    }}
    .hero-stat {{
        font-family: 'Source Serif Pro', Georgia, serif;
        font-size: 4rem;
        font-weight: 700;
        color: {APPLE};
        line-height: 1;
        margin: 0.5rem 0 0.4rem 0;
        letter-spacing: -0.03em;
    }}
    .hero-stat-cap {{
        color: {INK};
        font-size: 0.95rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }}

    /* ---------- Aisle label (small uppercase pill) ---------- */
    .aisle-marker {{
        display: inline-block;
        background: {PRODUCE};
        color: {CREAM};
        padding: 0.28rem 0.85rem;
        border-radius: 14px;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        font-weight: 600;
        margin-bottom: 0.6rem;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
    }}

    /* ---------- Callouts (pull-quote boxes) ---------- */
    .callout {{
        border-left: 4px solid {APPLE};
        background: #FFF8F0;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin: 1.25rem 0;
        color: {INK};
        font-size: 0.96rem;
        line-height: 1.55;
        font-family: 'Inter', sans-serif;
    }}
    .callout-good {{
        border-left-color: {LEAF};
        background: #F4F8EE;
    }}
    .callout-warn {{
        border-left-color: {PEACH};
        background: #FFF3EB;
    }}
    .callout b {{ color: {PRODUCE}; }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: {PAPER} !important;
        border-right: 1px solid #E5DFD3;
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 1rem;
    }}
    /* Sidebar inherits Inter from body; per-element rules below override
       to serif for the brand wordmark. */
    /* ---------- Sidebar brand block (logo treatment) ---------- */
    .sidebar-logo {{
        text-align: center;
        padding: 0.4rem 0 1.4rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #E5DFD3;
    }}
    .sidebar-brand-emoji {{
        font-size: 2.3rem;
        line-height: 1;
        display: block;
        margin-bottom: 0.25rem;
    }}
    .sidebar-brand {{
        font-family: 'Source Serif Pro', Georgia, serif !important;
        font-size: 1.85rem !important;
        color: {PRODUCE};
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.05rem 0;
        line-height: 1.05;
        display: block;
    }}
    .sidebar-brand-place {{
        font-family: 'Source Serif Pro', Georgia, serif !important;
        color: {APPLE};
        font-style: italic;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.05rem 0 0.55rem 0;
        display: block;
    }}
    .sidebar-tagline {{
        color: {BAG};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin: 0;
        font-weight: 600;
    }}
    .sidebar-group {{
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.68rem;
        color: {BAG};
        font-weight: 700;
        margin: 1.2rem 0 0.4rem 0;
        padding-left: 0.25rem;
    }}
    .sidebar-link {{
        display: block;
        padding: 0.42rem 0.6rem;
        margin: 0.05rem 0;
        color: {INK};
        text-decoration: none !important;
        border-radius: 4px;
        font-size: 0.88rem;
        line-height: 1.35;
        border-left: 2px solid transparent;
        transition: all 0.15s ease-in-out;
    }}
    .sidebar-link:hover {{
        background: {CREAM};
        color: {PRODUCE} !important;
        border-left-color: {LEAF};
        padding-left: 0.85rem;
    }}
    .sidebar-foot {{
        font-size: 0.72rem;
        color: {BAG};
        font-style: italic;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #E5DFD3;
        line-height: 1.5;
    }}

    /* ---------- Layout ---------- */
    .block-container {{
        padding-top: 1.5rem !important;
        max-width: 1180px !important;
    }}

    /* ---------- Streamlit elements ---------- */
    .stDataFrame, .stTable {{
        font-size: 0.9rem;
    }}
    .stMetric label {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.78rem !important;
        color: {BAG} !important;
        letter-spacing: 0.04em;
        font-weight: 600 !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Source Serif Pro', Georgia, serif !important;
        color: {PRODUCE} !important;
    }}
    .stButton > button {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 600;
        border-radius: 4px;
    }}
    .stMarkdown p, .stMarkdown li {{
        line-height: 1.65;
        color: {INK};
    }}
    /* Captions slightly muted */
    [data-testid="stCaptionContainer"] {{
        color: {BAG} !important;
        font-style: italic;
    }}

    /* Plotly charts — match background */
    .js-plotly-plot, .plotly {{
        background-color: transparent !important;
    }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    path = config.OUTPUTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_manifest() -> dict | None:
    path = config.OUTPUTS_DIR / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_resource(show_spinner=False)
def load_spoilage_model():
    path = config.MODELS_DIR / "spoilage_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def anchor(name: str) -> None:
    """Drop an HTML anchor so the sidebar can jump to it."""
    st.markdown(f"<a name='{name}'></a>", unsafe_allow_html=True)


def divider(emoji: str = "🛒") -> None:
    st.markdown(f"<div class='aisle-divider'>{emoji}</div>", unsafe_allow_html=True)


def aisle_label(text: str) -> None:
    st.markdown(f"<div class='aisle-marker'>{text}</div>", unsafe_allow_html=True)


def callout(text: str, variant: str = "neutral") -> None:
    klass = {"good": "callout callout-good",
             "warn": "callout callout-warn",
             "honest": "callout"}.get(variant, "callout")
    st.markdown(f"<div class='{klass}'>{text}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — table of contents (anchor links), grouped into chapters
# ─────────────────────────────────────────────────────────────────────────────
SIDEBAR_GROUPS = [
    ("Walking in", [
        ("welcome",   "🌽", "Welcome"),
        ("problem",   "🥬", "The problem"),
    ]),
    ("How it works", [
        ("approach",  "🛒", "Our approach"),
        ("datasets",  "📦", "The data"),
    ]),
    ("The results", [
        ("showdown",  "💰", "Strategy showdown"),
        ("aisles",    "🥩", "Aisle-by-aisle uplift"),
        ("expiry",    "⏰", "Sell-by-date logic"),
        ("register",  "💳", "Discount distribution"),
    ]),
    ("Stress & sensitivity", [
        ("stress",      "🚨", "Near-expiry stress"),
        ("sensitivity", "🔍", "Elasticity sensitivity"),
    ]),
    ("Interactive & detail", [
        ("whatif",   "🧪", "Try it yourself"),
        ("models",   "🔧", "Under the hood"),
    ]),
    ("Closing", [
        ("limits",   "⚠️", "Honest limits"),
        ("checkout", "🧾", "Final receipt"),
    ]),
]


def render_sidebar():
    with st.sidebar:
        # Logo block — emoji on top, two-line wordmark, course tagline
        st.markdown(
            "<div class='sidebar-logo'>"
            "<span class='sidebar-brand-emoji'>🥬</span>"
            "<span class='sidebar-brand'>Fresh Pricing</span>"
            "<span class='sidebar-brand-place'>in Missouri</span>"
            "<div class='sidebar-tagline'>FSE 570 · Capstone</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        for group_title, items in SIDEBAR_GROUPS:
            st.markdown(
                f"<div class='sidebar-group'>{group_title}</div>",
                unsafe_allow_html=True,
            )
            for slug, emoji, label in items:
                st.markdown(
                    f"<a class='sidebar-link' href='#{slug}'>"
                    f"<span style='display:inline-block; width:1.4em;'>{emoji}</span>"
                    f"{label}</a>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            "<div class='sidebar-foot'>"
            "Scroll the page to walk the store top-to-bottom, "
            "or click any aisle to jump."
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sections (rendered top-to-bottom on a single long page)
# ─────────────────────────────────────────────────────────────────────────────
def section_welcome():
    anchor("welcome")
    delta = load_csv("strategy_delta_table.csv")
    manifest = load_manifest()

    st.markdown(f"""
<div class='hero'>
  <div class='hero-title'>Fresh Pricing</div>
  <p class='hero-sub'>A walk through a data-driven grocery store that knows<br/>
    when to mark something down, and when to leave it alone.</p>
""", unsafe_allow_html=True)

    if delta is not None:
        try:
            dyn = float(delta.set_index("Strategy").loc["Dynamic", "Profit_USD"])
            base = float(delta.set_index("Strategy").loc["No Discount", "Profit_USD"])
            multiple = dyn / base if base else None
            waste_pct = float(delta.set_index("Strategy").loc["Dynamic", "Waste_Pct"])
            sellt = float(delta.set_index("Strategy").loc["Dynamic", "Sell_Through_Pct"])
        except Exception:
            multiple = None
            waste_pct = sellt = None

        if multiple is not None:
            st.markdown(f"""
  <div class='hero-stat'>{multiple:.1f}×</div>
  <div class='hero-stat-cap'>more profit than a no-discount baseline · 60-day out-of-sample test</div>
</div>
""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Profit (Dynamic)", f"${dyn:,.0f}",
                          delta=f"+${dyn - base:,.0f} vs baseline")
            with c2:
                st.metric("Waste rate", f"{waste_pct:.1f}%",
                          delta=f"{waste_pct - 33.5:+.1f} pp vs baseline",
                          delta_color="inverse")
            with c3:
                st.metric("Sell-through", f"{sellt:.0f}%",
                          delta=f"{sellt - 66:+.0f} pp vs baseline")

    st.markdown("""
This is what a grocery chain could actually deploy. For every SKU on every
day, the system gives one decision: how much, if anything, to mark this item
down. The decision pulls together a demand forecast, the customer's price
sensitivity, and the chance the item spoils before it sells. **Walk through
the aisles below** to see how it works, why it works, and where it doesn't.
""")


def section_problem():
    divider("🥬")
    anchor("problem")
    aisle_label("Aisle 1 · The Problem")
    st.markdown("## 10 to 15% of every grocery store walks straight into the dumpster")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
That's the industry baseline for perishable food spoilage, and it happens
because markdown decisions are usually **manual, ad-hoc, and reactive**.

Three failure modes show up over and over:

- **Too late.** The item spoils before anyone marks it down.
- **Too early.** Margin gets sacrificed on items that would have sold at full price.
- **Uniform.** A flat 20% off the whole shelf, regardless of expiry or demand.

The first two waste food. The third one is the interesting one: it actually
**destroys profit** versus doing nothing at all, and we'll show that in a
later aisle.

The right answer isn't more aggressive discounting. It's **targeted**
discounting. Deep cuts where spoilage risk is high *and* customers actually
respond to a lower price. No cuts at all where the shelf life is plenty and
people would have bought it anyway.
""")
    with c2:
        st.markdown(f"""
<div class='callout callout-warn'>
<b>Why this matters beyond the spreadsheet</b><br/><br/>
Food waste is the third-largest source of greenhouse gas emissions globally
(<i>Project Drawdown, 2020</i>). Reducing grocery shrinkage isn't only a
margin story. It's also a sustainability story.
</div>
""", unsafe_allow_html=True)


def section_approach():
    divider("🛒")
    anchor("approach")
    aisle_label("Aisle 2 · How we built it")
    st.markdown("## Five stages, from raw transactions to a daily markdown recommendation")

    st.markdown("""
The system is a chain of five stages. Each stage feeds the next. The whole
thing sits inside a strict train, validation, and holdout split, so when we
say a number is out-of-sample, it actually is.
""")

    stages = [
        ("1", "Demand forecasting",
         "LightGBM beat Prophet and Holt-Winters on a 30-day validation window. "
         "We picked the winner by **directional accuracy** (62% versus roughly "
         "50% coin-flip) because the optimizer needs to know whether demand is "
         "going up or down, not just hit the right average."),
        ("2", "Price elasticity",
         "Log-log regression on Dunnhumby with **store + week fixed effects**. "
         "Every department turned out to be price-inelastic, but the spread "
         "matters: Produce ( -0.23 ) is about four times more responsive than "
         "Meat ( -0.06 ). That alone tells us a flat markdown can't be right."),
        ("3", "Spoilage classifier",
         "Four classifiers compared: Logistic Regression, Random Forest, "
         "XGBoost, and LightGBM. LightGBM wins (AUC 0.85, Brier 0.15). The "
         "predicted probability is calibrated against the empirical waste "
         "fraction so it translates into expected wasted units, not just a yes-or-no."),
        ("4", "Optimizer",
         "For every SKU on every day, we score five candidate discounts "
         "(0, 10, 20, 30, 40 percent) and pick the one that **maximises "
         "expected profit** above a 5% margin floor. Items within five days of "
         "expiry use a stronger -2.0 elasticity to model bargain-hunter "
         "behaviour."),
        ("5", "60-day out-of-sample simulation",
         "Three strategies are compared on the held-out final 60 days: No "
         "Discount, Fixed 20%, and Dynamic. None of the models have ever seen "
         "this data."),
    ]
    for num, title, body in stages:
        c1, c2 = st.columns([1, 5])
        with c1:
            st.markdown(
                f"<div style='font-family: \"Source Serif Pro\", Georgia, serif; "
                f"font-size: 2.4rem; color:{LEAF}; font-weight:700; "
                f"text-align:center; line-height:1; margin-top:0.4rem;'>"
                f"{num}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(f"#### {title}")
            st.markdown(body)

    callout("""
<b>The discipline that makes the headline real:</b> every model is trained
only on data before the holdout window. The final 60 days stay sealed off
until stage 5. So the $1.99M profit number isn't a retrospective replay of
data the models already learned from. It's what they would have produced on
data they had never seen.
""", "good")


def section_datasets():
    divider("📦")
    anchor("datasets")
    aisle_label("Aisle 3 · The two suppliers")
    st.markdown("## Two public datasets, integrated at the category level")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
<div style='padding:1.2rem 1.4rem; background:{PAPER}; border-left:4px solid {LEAF};
            border-radius:4px;'>
<h4 style='margin-top:0;'>🛒 Dunnhumby, The Complete Journey</h4>
<b>Used for:</b> price elasticity<br/>
<b>Scale:</b> 2,500 households, 102 weeks, 2.4M product-store-week rows
<div style='margin-top:0.7rem;'><b>Anchor field:</b> regular shelf price equals</div>
<div style='font-family: ui-monospace, Menlo, Consolas, monospace;
     background: #FFF8EC; padding: 0.6rem 0.8rem; border-radius: 4px;
     margin-top: 0.4rem; font-size: 0.85rem; white-space: nowrap; overflow-x: auto;'>
(SALES_VALUE - RETAIL_DISC - COUPON_MATCH_DISC) / QUANTITY
</div>
<div style='margin-top:0.7rem;'><i>Stripping out promotional discounts gives us
the elasticity of base demand to base price.</i></div>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div style='padding:1.2rem 1.4rem; background:{PAPER}; border-left:4px solid {APPLE};
            border-radius:4px;'>
<h4 style='margin-top:0;'>🥬 Kaggle Perishable Goods Management</h4>
<b>Used for:</b> demand, spoilage, simulation<br/>
<b>Scale:</b> 100,000 SKU-day rows, 50 stores, 8 categories, 2 years
<div style='margin-top:0.7rem;'><b>Key fields:</b></div>
<div style='font-family: ui-monospace, Menlo, Consolas, monospace;
     background: #FFF8EC; padding: 0.6rem 0.8rem; border-radius: 4px;
     margin-top: 0.4rem; font-size: 0.82rem; line-height: 1.7;'>
days_until_expiry, shelf_life_days, units_wasted,<br/>
spoilage_risk, cost_price, base_price
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### How we connect the two")
    st.markdown("""
The datasets share **no common SKU identifier**. We can't say "this exact
yogurt in Dunnhumby is the same yogurt in the Kaggle data." So the join
happens at the category level. Each perishable category maps to its closest
Dunnhumby department, and the elasticity we learn from Dunnhumby is applied
to the matching category in the perishable optimizer.
""")
    cat_to_dept = pd.DataFrame([
        {"Perishable category": k, "Mapped to department": v}
        for k, v in config.CAT_TO_DEPT.items()
    ])
    st.dataframe(cat_to_dept, use_container_width=True, hide_index=True)
    st.caption(
        "Pharmaceuticals and Frozen Meals are excluded. Their shelf-life "
        "behaviour doesn't generalise to fresh perishables, and including them "
        "would skew the model. After exclusions, the working dataset has "
        "79,957 records across 8 categories."
    )


def section_showdown():
    divider("💰")
    anchor("showdown")
    aisle_label("Aisle 4 · Strategy showdown")
    st.markdown("## Three pricing rules walk into a 60-day holdout")

    delta = load_csv("strategy_delta_table.csv")
    if delta is None:
        st.warning("strategy_delta_table.csv not found. Run the pipeline first.")
        return

    show = st.multiselect(
        "Compare strategies (toggle to focus)",
        options=delta["Strategy"].tolist(),
        default=delta["Strategy"].tolist(),
        key="showdown_filter",
    )
    df = delta[delta["Strategy"].isin(show)].copy()

    # Build labels as DataFrame columns up-front so each bar gets the right
    # label. Using fig.update_traces(text=[...]) with a multi-color px.bar
    # is broken: plotly creates one trace per color category, then broadcasts
    # the single text list to every trace, causing all bars to display the
    # same (first) value.
    df = df.copy()
    df["lbl_profit"]  = df["Profit_USD"].apply(lambda v: f"${v:,.0f}")
    df["lbl_waste"]   = df["Waste_Pct"].apply(lambda v: f"{v:.1f}%")
    df["lbl_sellt"]   = df["Sell_Through_Pct"].apply(lambda v: f"{v:.0f}%")
    df["lbl_revenue"] = df["Revenue_USD"].apply(lambda v: f"${v:,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="Strategy", y="Profit_USD",
                     color="Strategy", color_discrete_map=STRAT_PALETTE,
                     text="lbl_profit",
                     title="Total profit (60 days, all SKUs)",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, height=380, yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(df, x="Strategy", y="Waste_Pct",
                     color="Strategy", color_discrete_map=STRAT_PALETTE,
                     text="lbl_waste",
                     title="Waste as % of inventory",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, height=380, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(df, x="Strategy", y="Sell_Through_Pct",
                     color="Strategy", color_discrete_map=STRAT_PALETTE,
                     text="lbl_sellt",
                     title="Sell-through rate", template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, height=380, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.bar(df, x="Strategy", y="Revenue_USD",
                     color="Strategy", color_discrete_map=STRAT_PALETTE,
                     text="lbl_revenue",
                     title="Total revenue", template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, height=380, yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)

    daily = load_csv("daily_profit_trend.csv")
    if daily is not None:
        daily["date"] = pd.to_datetime(daily["date"])
        long = daily.melt(id_vars="date", var_name="Strategy", value_name="Profit")
        long = long[long["Strategy"].isin(show)]
        fig = px.line(long, x="date", y="Profit", color="Strategy",
                      color_discrete_map=STRAT_PALETTE,
                      title="Daily profit across the 60-day holdout",
                      template=PLOTLY_TEMPLATE)
        fig.update_layout(height=360, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    callout("""
<b>The Fixed 20% number is the most interesting one on this page.</b> A
blanket 20% off actually <i>loses</i> about $70,000 in profit compared to
doing nothing at all. The reason is simple. The 20% rule discounts items
that would have sold at full price anyway, sacrificing margin on those, and
in exchange it only partially helps the items that actually needed help.
<b>Indiscriminate markdowns end up worse than no markdowns.</b> That single
comparison is the strongest argument here for picking discounts at the SKU
level instead of the shelf level.
""", "honest")


def section_aisles():
    divider("🥩")
    anchor("aisles")
    aisle_label("Aisle 5 · Aisle by aisle")
    st.markdown("## Where the optimizer wins big, and where it can't help")

    cat = load_csv("category_profit.csv")
    if cat is None:
        st.warning("category_profit.csv not found.")
        return

    cat["uplift"] = cat["Dynamic"] - cat["No Discount"]
    cat = cat.sort_values("uplift", ascending=False)

    pick = st.multiselect(
        "Filter categories",
        options=cat["category"].tolist(),
        default=cat["category"].tolist(),
        key="aisles_filter",
    )
    show = cat[cat["category"].isin(pick)]

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=show["category"], y=show["No Discount"],
                              name="No Discount",
                              marker_color=STRAT_PALETTE["No Discount"]))
        fig.add_trace(go.Bar(x=show["category"], y=show["Dynamic"],
                              name="Dynamic",
                              marker_color=STRAT_PALETTE["Dynamic"]))
        fig.update_layout(barmode="group", height=420,
                          title="Profit by category, baseline vs Dynamic",
                          template=PLOTLY_TEMPLATE, yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Bar(
            x=show["uplift"], y=show["category"], orientation="h",
            marker_color=[LEAF if v > 0 else APPLE for v in show["uplift"]],
            text=[f"${v:+,.0f}" for v in show["uplift"]], textposition="outside",
        ))
        fig.update_layout(height=420, title="Uplift from Dynamic",
                          template=PLOTLY_TEMPLATE, xaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(show.set_index("category").round(2), use_container_width=True)

    callout("""
<b>Honest read on Ready-to-Eat.</b> Even under the Dynamic strategy this
category still loses $158,000. The optimizer pulls it from a $431K loss up
to a $158K loss (so a $272K improvement), but it can't push the category
into the black. The reason is structural. Thin margins meet a high spoilage
rate, and no pricing rule can fix broken unit economics. In a real
deployment you'd either drop this category, renegotiate the supply terms,
or fix shelf-life management upstream before touching the pricing layer.
""", "honest")


def section_expiry():
    divider("⏰")
    anchor("expiry")
    aisle_label("Aisle 6 · The sell-by-date game")
    st.markdown("## Discount depth scales with how close the expiry date is")
    st.markdown("""
The most important behaviour we want from the optimizer is simple. If an
item is about to spoil, discount it. If it has plenty of shelf life left,
leave it at full price. Here's whether it actually does that.
""")

    bk = load_csv("near_expiry_buckets.csv")
    if bk is None:
        st.warning("near_expiry_buckets.csv not found.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bk["expiry_bucket"], y=bk["avg_discount_pct"],
        marker_color=APPLE, name="Avg discount",
        text=[f"{v:.1f}%" for v in bk["avg_discount_pct"]],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=bk["expiry_bucket"], y=bk["avg_profit_per_unit"],
        mode="lines+markers", yaxis="y2", name="Profit / unit",
        line=dict(color=LEAF, width=3),
    ))
    fig.update_layout(
        title="Optimizer behaviour by days until expiry",
        template=PLOTLY_TEMPLATE, height=420,
        yaxis=dict(title="Avg discount (%)"),
        yaxis2=dict(title="Profit / unit ($)", overlaying="y", side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bk.set_index("expiry_bucket"), use_container_width=True)

    callout("""
<b>Mechanism check passed.</b> Items with one or two days left get an average
23.1% discount, and roughly 76% of them are discounted at all. Items with
six or more days left get nothing. Profit per unit does drop at the deep
end of the discount range, because you're sacrificing margin to convert
soon-to-spoil inventory into actual revenue instead of waste. That's the
trade-off you want the system to make.
""", "good")


def section_register():
    divider("💳")
    anchor("register")
    aisle_label("Aisle 7 · At the register")
    st.markdown("## Most SKUs get no discount at all")

    dist = load_csv("discount_distribution.csv")
    if dist is None:
        st.warning("discount_distribution.csv not found.")
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(dist, x="discount", y="pct_of_skus",
                     title="Recommended discount distribution (1k-row sample)",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(marker_color=LEAF,
                          text=[f"{v:.1f}%" for v in dist["pct_of_skus"]],
                          textposition="outside")
        fig.update_layout(height=380, xaxis_tickformat=".0%",
                          xaxis_title="Discount", yaxis_title="% of SKUs")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("""
**More than half of all SKUs get zero discount.** The optimizer is
conservative by default. It only marks an item down when the combination of
spoilage risk and customer price sensitivity actually justifies the lost
margin.

That's the biggest design difference between this and a flat 20% rule. The
optimizer's discounts are earned, not sprayed across the shelf.
""")

    cat_disc = load_csv("category_mean_discount.csv")
    if cat_disc is not None:
        st.subheader("Mean recommended discount by category")
        fig = px.bar(cat_disc.sort_values("discount", ascending=False),
                     x="category", y="discount",
                     color="discount", color_continuous_scale=[CREAM, APPLE],
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(height=360, yaxis_title="Avg discount (%)",
                          showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def section_stress():
    divider("🚨")
    anchor("stress")
    aisle_label("Aisle 8 · Near-expiry stress test")
    st.markdown("## When everything is about to spoil at once")

    hs = load_csv("high_spoilage_kpi.csv")
    if hs is None:
        st.warning("high_spoilage_kpi.csv not found.")
        return

    if hs.columns[0] != "strategy":
        hs = hs.rename(columns={hs.columns[0]: "strategy"})
    hs["label"] = hs["strategy"].map({"no_discount": "No Discount",
                                       "fixed_20": "Fixed 20%",
                                       "dynamic": "Dynamic"})

    # Per-bar text labels as columns (see fix in section_showdown for the
    # explanation of why update_traces(text=[...]) doesn't work here).
    hs["lbl_profit"] = hs["Total_Profit"].apply(lambda v: f"${v:,.0f}")
    hs["lbl_waste"]  = hs["Total_Waste"].apply(lambda v: f"{v:,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(hs, x="label", y="Total_Profit", color="label",
                     color_discrete_map=STRAT_PALETTE,
                     text="lbl_profit",
                     title="Profit on near-expiry subset (≤3 days to expiry)",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=380, showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(hs, x="label", y="Total_Waste", color="label",
                     color_discrete_map=STRAT_PALETTE,
                     text="lbl_waste",
                     title="Units wasted on near-expiry subset",
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=380, showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(hs.set_index("label").drop(columns=["strategy"]),
                 use_container_width=True)

    callout("""
<b>Every strategy is unprofitable on this slice.</b> Heavy markdown plus
residual waste eats margin no matter what you do. But the relative gain
from the Dynamic strategy is biggest right here. It cuts losses by roughly
$1.1M versus the no-discount baseline on the near-expiry items alone. The
case for dynamic pricing is strongest exactly where the alternatives hurt
the most.
""", "warn")


def section_sensitivity():
    divider("🔍")
    anchor("sensitivity")
    aisle_label("Aisle 9 · What if we got elasticity wrong?")
    st.markdown("## Sensitivity analysis, and why the pretty version is misleading")

    es = load_csv("elasticity_sensitivity.csv")
    if es is None:
        st.warning("elasticity_sensitivity.csv not found.")
        return

    overall = es[es["discount"] == -1.0].copy()
    overall["scenario"] = pd.Categorical(
        overall["scenario"],
        categories=["Elast x 0.5", "Baseline", "Elast x 1.5"], ordered=True,
    )
    overall = overall.sort_values(["subset", "scenario"])

    subsets = overall["subset"].unique().tolist()
    cols = st.columns(len(subsets))
    for col, sub in zip(cols, subsets):
        with col:
            sub_df = overall[overall["subset"] == sub].copy()
            sub_df["lbl"] = sub_df["mean_profit"].apply(lambda v: f"${v:.2f}")
            label = ("Full sample, dominated by near-expiry rows that hit the override"
                     if sub == "ALL_ROWS"
                     else "Non-near-expiry rows only, which is the real test")
            fig = px.bar(sub_df, x="scenario", y="mean_profit",
                         title=label, color="scenario",
                         color_discrete_sequence=["#A0A0A0", LEAF, "#A0A0A0"],
                         text="lbl",
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=400, showlegend=False, xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    callout("""
<b>Read this one carefully.</b> On the full sample, mean profit barely
moves across 0.5x, 1.0x, and 1.5x elasticity. That looks like the optimizer
is robust to elasticity, but it isn't. The reason is that <b>42% of holdout
rows are near-expiry</b>, and those rows use the fixed -2.0 elasticity
override regardless of the multiplier we tweak.<br/><br/>
The honest test is the <b>non-near-expiry subset</b> on the right. If that
also barely moves under plus or minus 50%, then the regular elasticity is
doing limited work in the optimizer's decisions. That's a real finding
worth reporting, not the same thing as "stability." Don't oversell this
section in the writeup.
""", "honest")

    st.subheader("Discount distribution under each scenario")
    disc = es[es["discount"] >= 0].copy()
    disc["discount_pct"] = (disc["discount"] * 100).astype(int)
    fig = px.bar(
        disc, x="discount_pct", y="n_rows",
        color="scenario", facet_col="subset", barmode="group",
        labels={"discount_pct": "Discount (%)", "n_rows": "# of SKUs"},
        title="How elasticity changes shift the discount distribution",
        color_discrete_sequence=[STRAT_PALETTE["No Discount"], LEAF, APPLE],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def section_whatif():
    divider("🧪")
    anchor("whatif")
    aisle_label("Aisle 10 · Try it yourself")
    st.markdown("## Move the dials and re-run the optimizer with your assumptions")

    sample = load_csv("whatif_sample.csv")
    bundle = load_spoilage_model()
    fc_df = load_csv("sku_demand_forecasts.csv")

    if sample is None or bundle is None:
        st.error("""
The What-if explorer needs **`outputs/whatif_sample.csv`** and
**`models/spoilage_model.pkl`**. Run `python save_whatif_files.py` from the
`code/` folder, then refresh this page.
""")
        return

    spoilage_model = bundle["model"]
    train_cols = bundle["train_cols"]
    waste_frac = bundle["waste_frac"]

    if fc_df is not None:
        fc_df["date"] = pd.to_datetime(fc_df["date"])
        forecast_lookup = {(r["date"], r["category"]): r["forecast"]
                           for _, r in fc_df.iterrows()}
    else:
        forecast_lookup = {}

    sample = sample.copy()
    sample["transaction_date"] = pd.to_datetime(sample["transaction_date"])

    st.markdown("### Your assumptions")
    c1, c2, c3 = st.columns(3)
    with c1:
        elas_mult = st.slider(
            "Regular elasticity multiplier", 0.25, 2.0, 1.0, 0.25,
            help="Scales the per-department elasticity. The near-expiry override is unaffected.",
        )
    with c2:
        ne_thresh = st.slider(
            "Near-expiry threshold (days)", 1, 14, config.NEAR_EXPIRY_DAYS, 1,
            help="Items at or below this many days to expiry use the bargain-hunter elasticity.",
        )
    with c3:
        ne_elas = st.slider(
            "Near-expiry elasticity", -4.0, -1.0, config.NEAR_EXPIRY_ELAS, 0.25,
            help="Literature: -1.5 to -3.0. The default is -2.0.",
        )
    c4, c5 = st.columns(2)
    with c4:
        margin = st.slider("Margin floor (price ≥ cost ×)",
                           1.0, 1.20, 1.0 + config.MIN_MARGIN, 0.01)
    with c5:
        max_disc = st.slider("Maximum allowed discount (%)",
                             10, 60, 40, 10)
    grid = [round(d / 100, 2) for d in range(0, max_disc + 1, 10)]
    st.caption(f"Discount grid: {[f'{int(d * 100)}%' for d in grid]}")

    if st.button("▶ Re-run optimizer with these settings", type="primary"):
        with st.spinner("Re-running optimizer on the 500-row sample..."):
            from optimizer import batch_simulate

            base_elas = config.DEFAULT_ELASTICITY_MAP
            scaled = {k: v * elas_mult for k, v in base_elas.items()}

            old_grid = config.DISCOUNT_GRID
            old_thresh = config.NEAR_EXPIRY_DAYS
            old_ne_elas = config.NEAR_EXPIRY_ELAS
            old_margin = config.MIN_MARGIN
            try:
                config.DISCOUNT_GRID = grid
                config.NEAR_EXPIRY_DAYS = ne_thresh
                config.NEAR_EXPIRY_ELAS = ne_elas
                config.MIN_MARGIN = margin - 1.0
                user_run = batch_simulate(
                    sample, "dynamic",
                    spoilage_model=spoilage_model, train_cols=train_cols,
                    elasticity_map=scaled, waste_frac=waste_frac,
                    demand_forecasts=forecast_lookup,
                )
                config.DISCOUNT_GRID = [0.0, 0.10, 0.20, 0.30, 0.40]
                config.NEAR_EXPIRY_DAYS = 5
                config.NEAR_EXPIRY_ELAS = -2.0
                config.MIN_MARGIN = 0.05
                base_run = batch_simulate(
                    sample, "dynamic",
                    spoilage_model=spoilage_model, train_cols=train_cols,
                    elasticity_map=base_elas, waste_frac=waste_frac,
                    demand_forecasts=forecast_lookup,
                )
            finally:
                config.DISCOUNT_GRID = old_grid
                config.NEAR_EXPIRY_DAYS = old_thresh
                config.NEAR_EXPIRY_ELAS = old_ne_elas
                config.MIN_MARGIN = old_margin

        c1, c2, c3 = st.columns(3)
        c1.metric("Profit (your settings)",
                  f"${user_run['profit'].sum():,.0f}",
                  delta=f"${user_run['profit'].sum() - base_run['profit'].sum():,.0f} vs project baseline")
        c2.metric("Expected waste (units)",
                  f"{user_run['exp_waste'].sum():,.0f}",
                  delta=f"{user_run['exp_waste'].sum() - base_run['exp_waste'].sum():,.0f}",
                  delta_color="inverse")
        c3.metric("Avg discount applied",
                  f"{user_run['discount'].mean() * 100:.1f}%",
                  delta=f"{(user_run['discount'].mean() - base_run['discount'].mean()) * 100:+.1f} pp")

        st.subheader("Discount distribution, your settings vs the project baseline")
        comp = pd.concat([
            user_run.assign(scenario="Your settings"),
            base_run.assign(scenario="Project baseline"),
        ])
        comp["discount_pct"] = (comp["discount"] * 100).round().astype(int)
        cnt = comp.groupby(["scenario", "discount_pct"]).size().reset_index(name="n")
        fig = px.bar(cnt, x="discount_pct", y="n", color="scenario", barmode="group",
                     labels={"discount_pct": "Discount (%)", "n": "# of SKUs"},
                     color_discrete_sequence=[LEAF, STRAT_PALETTE["No Discount"]],
                     template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)


def section_models():
    divider("🔧")
    anchor("models")
    aisle_label("Aisle 11 · Under the hood")
    st.markdown("## Model selection details")
    st.markdown(
        "Click any panel below to expand it. These are the comparison tables "
        "that drove the model picks."
    )

    with st.expander("📊 Demand forecasting, 30-day validation comparison"):
        cmp_demand = load_csv("demand_model_comparison.csv")
        if cmp_demand is not None:
            st.dataframe(cmp_demand, use_container_width=True, hide_index=True)
            st.caption("Selection rule: highest directional accuracy, RMSE tiebreak.")
        else:
            st.warning("demand_model_comparison.csv not found.")

    with st.expander("🧫 Spoilage classifier, 4-way comparison"):
        cmp_spoil = load_csv("spoilage_model_comparison.csv")
        if cmp_spoil is not None:
            st.dataframe(cmp_spoil, use_container_width=True, hide_index=True)
            st.caption("Selection rule: highest ROC-AUC; Brier tiebreak within 0.005 AUC. "
                       "Brier matters because the optimizer uses raw probabilities.")
        else:
            st.warning("spoilage_model_comparison.csv not found.")

    with st.expander("📈 Department elasticities and out-of-sample validation"):
        elas = load_csv("elasticity_estimates.csv")
        if elas is not None:
            st.dataframe(elas, use_container_width=True, hide_index=True)
            st.caption(
                "A negative OOS R squared looks alarming but it isn't a broken "
                "model. The out-of-sample test deliberately strips out the "
                "store and week fixed effects (those are unknown for future "
                "weeks), leaving only price as the predictor. What we're "
                "validating is whether the elasticity coefficient itself is "
                "stable across time windows. It is."
            )
        else:
            st.warning("elasticity_estimates.csv not found.")


def section_limits():
    divider("⚠️")
    anchor("limits")
    aisle_label("Aisle 12 · The honest limits")
    st.markdown("## What this analysis doesn't claim")

    st.markdown("""
A capstone is judged as much on what it doesn't claim as on what it does.
There are five real holes in this analysis. We're calling them out plainly
instead of burying them.
""")

    items = [
        ("1.  Elasticity transfer",
         "The elasticities we use come from Dunnhumby, which captures normal "
         "everyday grocery shopping. But a customer responding to a 30% off "
         "sticker on something that expires tomorrow is making a different "
         "kind of decision than one filling their weekly basket. We handle "
         "this with a -2.0 near-expiry override, informed by published "
         "markdown literature. That number is plausible, but **it isn't "
         "estimated from our own data**. A real deployment would need a "
         "small A/B markdown experiment to calibrate it properly."),
        ("2.  Ready-to-Eat is structurally negative",
         "Even under Dynamic pricing this category loses $158,000. The "
         "optimizer recovers $272,000 versus the no-discount baseline, but "
         "it can't push the category into the black. The right reading "
         "isn't that the model failed. It's that the unit economics are "
         "broken. The fix has to be upstream: margin renegotiation, cold "
         "chain, or shelf-life management. Pricing alone can't get there."),
        ("3.  The spoilage target is binary",
         "We predict whether any waste happens, not how much. We patch "
         "this by multiplying the predicted probability by an empirical "
         "waste fraction (0.1727) to translate it into expected wasted "
         "units. That's a reasonable workaround, but for high-inventory "
         "SKUs the difference between 5 wasted units and 200 is enormous. "
         "A regression model that predicts the waste fraction directly "
         "would be more precise. Left for future work."),
        ("4.  The elasticity sensitivity test looks too clean",
         "Under plus or minus 50% elasticity on the full sample, the "
         "optimizer's profit barely moves. That's mostly mechanical: 42% "
         "of rows hit the -2.0 override which doesn't scale with the "
         "multiplier. The honest test is on the non-near-expiry subset, "
         "and we show both. The report should not present the full-sample "
         "result as proof of stability."),
        ("5.  Demand is a point forecast, not probabilistic",
         "The optimizer treats the LightGBM forecast as a known number. "
         "In a real deployment, forecast uncertainty matters: a high-"
         "variance forecast should pull the optimizer toward more "
         "conservative discounts. We don't model that here."),
    ]
    for title, body in items:
        st.markdown(f"#### {title}")
        st.markdown(body)


def section_checkout():
    divider("🧾")
    anchor("checkout")
    aisle_label("Aisle 13 · The final receipt")
    st.markdown("## Checkout: what we showed and what comes next")

    delta = load_csv("strategy_delta_table.csv")

    st.markdown("""
### What the simulation showed

On a genuine 60-day out-of-sample holdout, the per-SKU per-day pricing
optimizer (demand forecast plus department elasticity plus calibrated
spoilage probability):

- Delivered roughly **$2M in profit, versus $420K under the no-discount
  baseline**. About 4.8 times more.
- Cut expected waste roughly **in half**.
- Raised sell-through from **66% to 79%**.

The most important single number on the page is actually the Fixed 20%
result: **a flat 20% discount underperforms doing nothing at all** ($349K
versus $420K). That comparison is the strongest argument here for picking
discounts at the SKU level instead of spraying them across the shelf.

### What we didn't show

A few things that would matter for a real deployment but aren't in this
project:

- Whether these gains hold in a real store. This is a simulation. The next
  step is a small field pilot in a handful of stores.
- Whether the 17.27% empirical waste fraction generalises across store
  formats, regions, or seasons.
- Whether the -2.0 near-expiry elasticity is right. The literature puts it
  somewhere between -1.5 and -3.0, and we picked the midpoint.

### What a real deployment would look like

A daily batch job. For every SKU in every store: pull yesterday's
inventory and remaining shelf life, call the demand forecaster, query the
spoilage model at each candidate price, run the discount grid, output one
recommended markdown. Add operational guardrails on top, like a cap on the
fraction of SKUs that can be discounted at once, or executive override on
premium items. Then A/B test the new policy against the current ad-hoc
markdown rules on a handful of stores before rolling it out chain-wide.
""")

    if delta is not None:
        st.subheader("Final scoreboard")
        st.dataframe(delta.set_index("Strategy"), use_container_width=True)

    st.markdown("---")
    st.markdown(f"""
<div style='text-align:center; color:{BAG}; font-style:italic; padding:1.5rem;'>
FSE 570 Capstone · Team Missouri<br/>
Bhavana Reddy Pasula · Sneha Nannapaneni · Moksha Smruthi Morapakula ·<br/>
Sree Padma Priya Abburi · Srujana Rachamalla
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main — render everything top-to-bottom
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    section_welcome()
    section_problem()
    section_approach()
    section_datasets()
    section_showdown()
    section_aisles()
    section_expiry()
    section_register()
    section_stress()
    section_sensitivity()
    section_whatif()
    section_models()
    section_limits()
    section_checkout()


if __name__ == "__main__":
    main()
