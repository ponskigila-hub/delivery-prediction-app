"""
Delivery Time Prediction — Manifest Terminal
A Streamlit app for predicting parcel delivery durations, styled after a
cargo waybill / customs manifest rather than a generic SaaS dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# PAGE CONFIG — MUST BE FIRST STREAMLIT CALL
# =============================================================================
st.set_page_config(
    page_title="Delivery Manifest",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DESIGN SYSTEM
# -----------------------------------------------------------------------------
# Subject: parcel logistics. The visual language borrows from air waybills,
# customs declarations and warehouse manifests — hairline-ruled forms,
# tracking-code monospace, a stamped "cleared" moment for the prediction
# result — instead of glassy gradient cards.
#
# Color   ink #12141A (page), panel #191C24, paper #F2ECDD (manifest sheet),
#         line #2B3040, accent #FF6A39 (stamp orange), accent-2 #E8C468
#         (manifest tan), text #E7E4DA, text-dim #8992A3, ok #4FAE7C,
#         warn #E2574C
# Type    Oswald (condensed, stenciled) for headings/labels,
#         IBM Plex Mono for tracking numbers, codes and metrics,
#         Inter for body copy
# Layout  hairline-ruled "form" sections, numbered manifest steps in the
#         sidebar (the pipeline genuinely is sequential), a single bold
#         moment: the prediction rendered as a torn-off shipping stub
# =============================================================================
MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink: #12141a;
    --panel: #191c24;
    --panel-2: #1f2330;
    --paper: #f2ecdd;
    --paper-dim: #d9d2bd;
    --line: #2b3040;
    --line-soft: #22262f;
    --text: #e7e4da;
    --text-dim: #8992a3;
    --accent: #ff6a39;
    --accent-soft: rgba(255, 106, 57, 0.14);
    --tan: #e8c468;
    --ok: #4fae7c;
    --warn: #e2574c;
}

/* ----- BASE ----- */
.stApp { background: var(--ink); }
.main > div { padding-top: 1rem; }
.block-container { padding-bottom: 3rem; max-width: 1320px; }

h1, h2, h3, h4 {
    font-family: 'Oswald', sans-serif;
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}
p, li, label, .stMarkdown, span { font-family: 'Inter', sans-serif; color: var(--text-dim); }
code, .stCode { font-family: 'IBM Plex Mono', monospace !important; }

/* ----- WAYBILL HEADER STRIP ----- */
.waybill-header {
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.6rem;
    background:
        repeating-linear-gradient(90deg, var(--panel) 0px, var(--panel) 3px, var(--panel-2) 3px, var(--panel-2) 5px) top / 100% 4px no-repeat,
        var(--ink);
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.waybill-header .wb-title {
    font-family: 'Oswald', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
}
.waybill-header .wb-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: var(--text-dim);
    margin-top: 0.2rem;
}
.waybill-header .wb-code {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    letter-spacing: 0.04em;
    align-self: flex-start;
}

/* ----- MANIFEST TAGS (metric cards) ----- */
.tag-row { display: flex; gap: 0.85rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
.manifest-tag {
    flex: 1 1 160px;
    position: relative;
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    padding: 0.85rem 1rem 0.75rem 1rem;
}
.manifest-tag .tag-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    text-transform: uppercase;
}
.manifest-tag .tag-value {
    font-family: 'Oswald', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.25;
}
.manifest-tag .tag-value .unit {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 400;
    color: var(--text-dim);
}
.manifest-tag .tag-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: var(--text-dim); margin-top: 0.15rem; }

/* ----- FORM SECTION (replaces bento cards) ----- */
.form-section {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 1.3rem 1.4rem;
}
.form-section h4 { margin-top: 0; }

/* ----- SIDEBAR AS MANIFEST LEDGER ----- */
section[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--line);
}
.ledger-brand { padding: 1rem 0 0.4rem 0.1rem; border-bottom: 1px solid var(--line); margin-bottom: 0.6rem; }
.ledger-brand .lb-mark { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--accent); letter-spacing: 0.1em; }
.ledger-brand h1 { font-size: 1.35rem; margin: 0.15rem 0 0 0; }

.stRadio [role="radiogroup"] { gap: 0.15rem; }
.stRadio label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.86rem !important;
    padding: 0.55rem 0.6rem !important;
    border-left: 2px solid transparent;
    color: var(--text-dim) !important;
    background: transparent !important;
    transition: border-color 0.15s ease, color 0.15s ease;
}
.stRadio label:hover { color: var(--text) !important; border-left-color: var(--line); }
.stRadio label[data-selected="true"] {
    color: var(--accent) !important;
    border-left-color: var(--accent);
    background: var(--accent-soft) !important;
}

.ledger-status {
    border: 1px solid var(--line);
    padding: 0.6rem 0.75rem;
    margin-top: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
}
.ledger-status .ls-label { font-size: 0.62rem; letter-spacing: 0.08em; color: var(--text-dim); text-transform: uppercase; }
.ledger-status .ls-value { font-size: 0.84rem; color: var(--text); margin-top: 0.1rem; }
.stamp-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
.stamp-dot.cleared { background: var(--ok); }
.stamp-dot.void { background: var(--warn); }

/* ----- BUTTONS ----- */
.stButton button {
    background: var(--accent) !important;
    color: #16130f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Oswald', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    padding: 0.55rem 1.4rem !important;
    box-shadow: none !important;
    transition: filter 0.15s ease;
}
.stButton button:hover { filter: brightness(1.08); transform: none; }
.stButton button:active { filter: brightness(0.94); }

/* ----- FORMS ----- */
div[data-testid="stForm"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 0;
    padding: 1.3rem 1.4rem;
}

/* ----- TABLES ----- */
.stDataFrame { border: 1px solid var(--line) !important; }

/* ----- TABS ----- */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--line); background: transparent; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--text-dim) !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ----- EXPANDER ----- */
.streamlit-expanderHeader {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--text-dim) !important;
}
.streamlit-expanderContent { background: var(--panel-2) !important; border: 1px solid var(--line) !important; border-top: none !important; }

/* ----- INPUTS ----- */
.stSelectbox label, .stNumberInput label, .stSlider label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
}

/* ----- ALERTS ----- */
.stAlert { border-radius: 0 !important; border: 1px solid var(--line) !important; }

/* ----- THE STUB (prediction result — the one bold moment) ----- */
.stub-wrap { display: flex; justify-content: center; margin: 0.5rem 0 1rem 0; }
.stub {
    background: var(--paper);
    color: #1c1712;
    width: 100%;
    max-width: 560px;
    position: relative;
    padding: 2rem 2rem 1.6rem 2rem;
    border-top: 1px dashed #b8ae90;
}
.stub::before {
    content: '';
    position: absolute;
    top: -1px; left: 0; right: 0;
    height: 0;
    border-top: 3px dashed transparent;
}
.stub .stub-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: #7a7157;
    display: flex;
    justify-content: space-between;
}
.stub .stub-days {
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    font-size: 4.2rem;
    line-height: 1.05;
    margin-top: 0.35rem;
    color: #1c1712;
}
.stub .stub-days .unit { font-size: 1.6rem; font-weight: 500; color: #7a7157; }
.stub .stub-ci { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #6a6248; margin-top: 0.2rem; }
.stub .stub-stamp {
    position: absolute;
    top: 1.4rem; right: 1.8rem;
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.06em;
    color: var(--ok);
    border: 2px solid var(--ok);
    padding: 0.15rem 0.6rem;
    transform: rotate(6deg);
    opacity: 0.85;
}
.stub .stub-fields {
    margin-top: 1.1rem;
    padding-top: 0.9rem;
    border-top: 1px solid #cfc6a8;
    display: flex;
    gap: 1.6rem;
    flex-wrap: wrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
}
.stub .stub-fields div span { display: block; }
.stub .stub-fields .f-label { color: #8a8163; font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; }
.stub .stub-fields .f-value { color: #1c1712; font-weight: 600; margin-top: 0.1rem; }

/* ----- MISC ----- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent !important; }
hr.rule { border: none; border-top: 1px solid var(--line); margin: 1.6rem 0; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--line); }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
"""
st.markdown(MAIN_CSS, unsafe_allow_html=True)

# =============================================================================
# MATPLOTLIB THEME (matches the manifest palette)
# =============================================================================
plt.rcParams.update({
    "axes.facecolor": "#191c24",
    "figure.facecolor": "#191c24",
    "axes.edgecolor": "#2b3040",
    "axes.labelcolor": "#8992a3",
    "xtick.color": "#8992a3",
    "ytick.color": "#8992a3",
    "text.color": "#e7e4da",
    "grid.color": "#22262f",
    "grid.alpha": 0.6,
    "legend.facecolor": "#191c24",
    "legend.edgecolor": "#2b3040",
    "font.family": "monospace",
})
sns.set_style("darkgrid")
ACCENT = "#ff6a39"
ACCENT_2 = "#e8c468"
OK = "#4fae7c"
WARN = "#e2574c"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def waybill_header(title, subtitle, code):
    st.markdown(f"""
    <div class="waybill-header">
        <div>
            <div class="wb-title">{title}</div>
            <div class="wb-sub">{subtitle}</div>
        </div>
        <div class="wb-code">{code}</div>
    </div>
    """, unsafe_allow_html=True)

def render_tag(label, value, unit="", sub=""):
    unit_html = f'<span class="unit"> {unit}</span>' if unit else ""
    sub_html = f'<div class="tag-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="manifest-tag">
        <div class="tag-label">{label}</div>
        <div class="tag-value">{value}{unit_html}</div>
        {sub_html}
    </div>
    """

def render_tag_row(tags):
    html = '<div class="tag-row">' + "".join(render_tag(**t) for t in tags) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def section_title(title):
    st.markdown(f'<h4 style="margin: 1.4rem 0 0.6rem 0;">{title}</h4>', unsafe_allow_html=True)

def rule():
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & PREPROCESSING (CACHED)
# =============================================================================
@st.cache_data
def load_data():
    """Load the dataset (fallback to synthetic data if file not found)."""
    try:
        df = pd.read_csv("olist_orders_dataset.csv")
    except FileNotFoundError:
        np.random.seed(42)
        n = 500
        dates = pd.date_range(start="2023-01-01", periods=n)
        df = pd.DataFrame({
            "order_purchase_timestamp": dates,
            "order_delivered_customer_date": dates + pd.to_timedelta(np.random.randint(2, 20, n), unit="d"),
            "order_estimated_delivery_date": dates + pd.to_timedelta(np.random.randint(5, 25, n), unit="d")
        })
    return df

@st.cache_data
def preprocess_data(df):
    """Clean and engineer features."""
    df = df.copy()
    df = df.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])

    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])

    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday

    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["estimated_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days

    df["delivery_days"] = df["delivery_days"].clip(lower=0, upper=60)
    df["estimated_days"] = df["estimated_days"].clip(lower=1, upper=60)

    return df

with st.spinner("Loading manifest data..."):
    raw_df = load_data()
    df = preprocess_data(raw_df)

FEATURES = ["purchase_hour", "purchase_day", "purchase_month", "purchase_weekday", "estimated_days"]
TARGET = "delivery_days"

# =============================================================================
# SESSION STATE INIT
# =============================================================================
for key in ["model", "scaler", "model_name", "trained", "metrics"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================================================================
# SIDEBAR — MANIFEST LEDGER (numbered because the pipeline is a real sequence)
# =============================================================================
STEPS = [
    ("01", "Home"),
    ("02", "EDA"),
    ("03", "Preprocessing"),
    ("04", "Train model"),
    ("05", "Evaluation"),
    ("06", "Predict"),
]

with st.sidebar:
    st.markdown("""
    <div class="ledger-brand">
        <div class="lb-mark">NO. 2941-DX &middot; CARGO MANIFEST</div>
        <h1>Delivery Terminal</h1>
    </div>
    """, unsafe_allow_html=True)

    page_label = st.radio(
        "Navigation",
        [f"{num} — {name}" for num, name in STEPS],
        label_visibility="collapsed",
    )
    page = page_label.split("— ")[1]

    if st.session_state.trained:
        model_dot, model_state, model_name = "cleared", "Trained", (st.session_state.model_name or "Model")
    else:
        model_dot, model_state, model_name = "void", "Not trained", "—"

    st.markdown(f"""
    <div class="ledger-status">
        <div class="ls-label">Dataset</div>
        <div class="ls-value">{len(df):,} rows &middot; {len(FEATURES)} fields</div>
    </div>
    <div class="ledger-status">
        <div class="ls-label">Model</div>
        <div class="ls-value"><span class="stamp-dot {model_dot}"></span>{model_name} &middot; {model_state}</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE ROUTING
# =============================================================================

# ---- HOME ----
if page == "Home":
    waybill_header("Delivery Time Prediction", "Predict parcel delivery duration from purchase and carrier data", "REF / DTP-001")

    render_tag_row([
        {"label": "Total orders", "value": f"{len(df):,}"},
        {"label": "Avg delivery", "value": f"{df[TARGET].mean():.1f}", "unit": "days"},
        {"label": "Max delivery", "value": f"{int(df[TARGET].max())}", "unit": "days"},
        {"label": "Std dev", "value": f"{df[TARGET].std():.1f}", "unit": "days"},
    ])

    rule()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        section_title("Data preview")
        st.dataframe(df.head(8), use_container_width=True)

    with col_right:
        section_title("Quick stats")
        st.markdown(f"""
        <div class="form-section">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.9rem; font-family: 'IBM Plex Mono', monospace;">
                <div><span style="color: var(--text-dim); font-size: 0.72rem;">MIN</span><br><span style="color: var(--text); font-weight: 600;">{int(df[TARGET].min())} days</span></div>
                <div><span style="color: var(--text-dim); font-size: 0.72rem;">MAX</span><br><span style="color: var(--text); font-weight: 600;">{int(df[TARGET].max())} days</span></div>
                <div><span style="color: var(--text-dim); font-size: 0.72rem;">MEDIAN</span><br><span style="color: var(--text); font-weight: 600;">{int(df[TARGET].median())} days</span></div>
                <div><span style="color: var(--text-dim); font-size: 0.72rem;">Q1–Q3</span><br><span style="color: var(--text); font-weight: 600;">{int(df[TARGET].quantile(0.25))}–{int(df[TARGET].quantile(0.75))} days</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---- EDA ----
elif page == "EDA":
    waybill_header("Exploratory Data Analysis", "Distributions, correlations and dataset profile", "REF / DTP-002")

    render_tag_row([
        {"label": "Total orders", "value": f"{len(df):,}"},
        {"label": "Features", "value": len(FEATURES)},
        {"label": "Avg delivery", "value": f"{df[TARGET].mean():.1f}", "unit": "days"},
        {"label": "Missing values", "value": f"{df.isnull().sum().sum():,}"},
    ])

    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Profile"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            section_title("Delivery days distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[TARGET], kde=True, color=ACCENT, ax=ax, bins=30)
            ax.set_xlabel("Delivery days")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()

        with col2:
            feat = st.selectbox("Select feature", FEATURES)
            section_title(f"{feat} distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[feat], kde=True, color=ACCENT_2, ax=ax, bins=20)
            ax.set_xlabel(feat)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()

        section_title("Boxplots")
        cols = st.columns(3)
        for i, feat in enumerate(FEATURES[:3]):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.boxplot(y=df[feat], color=ACCENT, ax=ax)
                ax.set_ylabel(feat)
                st.pyplot(fig)
                plt.close()

    with tab2:
        section_title("Feature correlation matrix")
        corr_df = df[FEATURES + [TARGET]].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, cmap="RdGy_r", center=0, fmt=".2f",
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        st.pyplot(fig)
        plt.close()

        section_title("Top correlations with target")
        corr_with_target = corr_df[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [OK if c > 0 else WARN for c in corr_with_target.values]
        ax.barh(corr_with_target.index, corr_with_target.values, color=colors)
        ax.set_xlabel("Correlation with delivery days")
        ax.axvline(0, color="#8992a3", linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close()

    with tab3:
        section_title("Data types")
        st.dataframe(df.dtypes.astype(str), use_container_width=True)
        section_title("Statistical summary")
        st.dataframe(df.describe(), use_container_width=True)

# ---- PREPROCESSING ----
elif page == "Preprocessing":
    waybill_header("Data Preprocessing", "Configure and run the preprocessing pipeline", "REF / DTP-003")

    with st.form("preprocessing_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05, format="%.2f")
        with col2:
            random_state = st.number_input("Random seed", 1, 999, 42)
        with col3:
            scaler_option = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])

        submitted = st.form_submit_button("Run preprocessing", use_container_width=True)

    if submitted:
        with st.spinner("Preprocessing data..."):
            X = df[FEATURES]
            y = df[TARGET]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            scaler = StandardScaler() if scaler_option == "StandardScaler" else MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            with open("scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

            st.success(f"Preprocessing complete — scaler saved as scaler.pkl ({scaler_option})")

            render_tag_row([
                {"label": "Training set", "value": X_train_scaled.shape[0], "unit": "rows"},
                {"label": "Test set", "value": X_test_scaled.shape[0], "unit": "rows"},
                {"label": "Features", "value": X_train_scaled.shape[1], "unit": "cols"},
                {"label": "Scaler", "value": scaler_option},
            ])

            with st.expander("Preview scaled data"):
                preview_df = pd.DataFrame(X_train_scaled[:5], columns=FEATURES)
                st.dataframe(preview_df, use_container_width=True)

# ---- TRAIN MODEL ----
elif page == "Train model":
    waybill_header("Model Training", "Fit a regression model on the preprocessed data", "REF / DTP-004")

    model_options = {
        "Linear Regression": LinearRegression(),
        "Random Forest (50 trees)": RandomForestRegressor(n_estimators=50, random_state=42),
        "Random Forest (100 trees)": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR (RBF)": SVR(kernel="rbf"),
        "SVR (Linear)": SVR(kernel="linear"),
    }

    selected_model = st.selectbox("Select algorithm", list(model_options.keys()))

    col1, col2 = st.columns(2)
    with col1:
        train_btn = st.button("Train model", use_container_width=True, type="primary")
    with col2:
        if st.button("Clear model", use_container_width=True):
            st.session_state.model = None
            st.session_state.scaler = None
            st.session_state.model_name = None
            st.session_state.trained = False
            st.session_state.metrics = None
            st.rerun()

    if train_btn:
        if not os.path.exists("scaler.pkl"):
            st.error("Run Preprocessing first to generate scaler.pkl")
        else:
            with st.spinner(f"Training {selected_model}..."):
                X = df[FEATURES]
                y = df[TARGET]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                with open("scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)

                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = model_options[selected_model]
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                metrics = {
                    "R2": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                }

                with open("MainModel.pkl", "wb") as f:
                    pickle.dump(model, f)

                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.model_name = selected_model
                st.session_state.trained = True
                st.session_state.metrics = metrics

                st.success(f"{selected_model} trained")

                render_tag_row([
                    {"label": "R2 score", "value": f"{metrics['R2']:.4f}"},
                    {"label": "MAE", "value": f"{metrics['MAE']:.3f}", "unit": "days"},
                    {"label": "RMSE", "value": f"{metrics['RMSE']:.3f}", "unit": "days"},
                ])

                with st.expander("Model details"):
                    if hasattr(model, "get_params"):
                        st.json(model.get_params())
                    st.write("Model type:", type(model).__name__)

# ---- EVALUATION ----
elif page == "Evaluation":
    waybill_header("Model Evaluation", "Assess model performance on the held-out test set", "REF / DTP-005")

    if st.session_state.trained and st.session_state.model is not None:
        model = st.session_state.model
        scaler = st.session_state.scaler

        X = df[FEATURES]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        metrics = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

        render_tag_row([
            {"label": "R2 score", "value": f"{metrics['R2']:.4f}"},
            {"label": "MAE", "value": f"{metrics['MAE']:.3f}", "unit": "days"},
            {"label": "MSE", "value": f"{metrics['MSE']:.3f}", "unit": "days sq."},
            {"label": "RMSE", "value": f"{metrics['RMSE']:.3f}", "unit": "days"},
        ])

        rule()

        col1, col2 = st.columns(2)
        with col1:
            section_title("Actual vs predicted")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color=ACCENT, ax=ax)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], color="#8992a3", linestyle="--", lw=1.5, alpha=0.8)
            ax.set_xlabel("Actual delivery days")
            ax.set_ylabel("Predicted delivery days")
            st.pyplot(fig)
            plt.close()

        with col2:
            section_title("Residuals distribution")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.histplot(residuals, kde=True, color=ACCENT_2, ax=ax, bins=25)
            ax.axvline(0, color=WARN, linestyle="--", alpha=0.7)
            ax.set_xlabel("Residual (actual - predicted)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()

        with st.expander("Full metrics table"):
            st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}), use_container_width=True)

    else:
        st.warning("No trained model on file. Go to Train model and fit one first.")

# ---- PREDICT ----
else:  # "Predict"
    waybill_header("Make a Prediction", "Enter order details to estimate delivery time", "REF / DTP-006")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            purchase_hour = st.slider("Purchase hour (0–23)", 0, 23, 12)
            purchase_day = st.number_input("Purchase day (1–31)", 1, 31, 15)
            purchase_weekday = st.selectbox("Purchase weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=2)
            weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
            purchase_weekday_val = weekday_map[purchase_weekday]

        with col2:
            purchase_month = st.selectbox("Purchase month", list(range(1, 13)), index=5)
            estimated_days = st.number_input("Carrier estimated days", 1, 60, 10)

        submitted = st.form_submit_button("Predict delivery time", use_container_width=True, type="primary")

    if submitted:
        if not st.session_state.trained or st.session_state.model is None:
            st.error("No trained model on file. Train a model first.")
        else:
            model = st.session_state.model
            scaler = st.session_state.scaler

            input_data = np.array([[
                purchase_hour,
                purchase_day,
                purchase_month,
                purchase_weekday_val,
                estimated_days
            ]])

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction = max(0, prediction)

            if st.session_state.metrics and "RMSE" in st.session_state.metrics:
                rmse = st.session_state.metrics["RMSE"]
                ci_lower = max(0, prediction - 1.96 * rmse)
                ci_upper = prediction + 1.96 * rmse
            else:
                ci_lower = max(0, prediction - 3)
                ci_upper = prediction + 3

            st.markdown(f"""
            <div class="stub-wrap">
                <div class="stub">
                    <div class="stub-eyebrow">
                        <span>ESTIMATED TRANSIT TIME</span>
                        <span>{purchase_day:02d}/{purchase_month:02d}, {purchase_hour:02d}:00</span>
                    </div>
                    <div class="stub-stamp">CLEARED</div>
                    <div class="stub-days">{prediction:.1f}<span class="unit"> days</span></div>
                    <div class="stub-ci">95% CI &nbsp;{ci_lower:.1f}–{ci_upper:.1f} days</div>
                    <div class="stub-fields">
                        <div><span class="f-label">Weekday</span><span class="f-value">{purchase_weekday}</span></div>
                        <div><span class="f-label">Carrier est.</span><span class="f-value">{estimated_days} days</span></div>
                        <div><span class="f-label">Model</span><span class="f-value">{st.session_state.model_name}</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Input summary"):
                st.json({
                    "purchase_hour": purchase_hour,
                    "purchase_day": purchase_day,
                    "purchase_month": purchase_month,
                    "purchase_weekday": purchase_weekday,
                    "estimated_days": estimated_days,
                    "prediction": round(prediction, 2),
                })

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div style="text-align: center; padding: 2rem 0 0.5rem 0; border-top: 1px solid var(--line); margin-top: 2rem;">
    <span style="font-family: 'IBM Plex Mono', monospace; color: #545a68; font-size: 0.72rem; letter-spacing: 0.04em;">
        DELIVERY MANIFEST TERMINAL — BUILT WITH STREAMLIT
    </span>
</div>
""", unsafe_allow_html=True)
