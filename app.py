"""
📦 Delivery Time Prediction App
A complete, dark-themed Streamlit application for predicting delivery durations.
Inspired by bento-style UI/UX with custom CSS, interactive visualizations,
and a full ML pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from datetime import datetime

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
    page_title="📦 Delivery Time Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — FULL DARK THEME WITH BENTO STYLE
# =============================================================================
MAIN_CSS = """
<style>
    /* ----- GLOBAL RESET & BASE ----- */
    .main > div {
        padding-top: 1.5rem;
    }
    .block-container {
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    .stApp {
        background: linear-gradient(145deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* ----- TYPOGRAPHY ----- */
    h1, h2, h3, h4, h5, h6 {
        color: #e8e8f0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, li, label, .stMarkdown {
        color: #b8b8d0 !important;
    }
    
    /* ----- BENTO CARDS ----- */
    .bento-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem 1.25rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .bento-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #4f8cf7, #7c5cfc, #4f8cf7);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
        opacity: 0.6;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .bento-card:hover {
        transform: translateY(-4px);
        border-color: rgba(79, 140, 247, 0.25);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
    }
    
    /* ----- METRIC CARDS (inside bento) ----- */
    .metric-value-lg {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f8cf7, #7c5cfc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .metric-value-lg .unit {
        font-size: 1.2rem;
        -webkit-text-fill-color: #8888bb;
        background: none;
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8888bb;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #666699;
        margin-top: 0.25rem;
    }
    
    /* ----- PREDICTION RESULT CARD ----- */
    .result-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.12), rgba(39, 174, 96, 0.04));
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 40px rgba(46, 204, 113, 0.08);
    }
    .result-value {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    .result-label {
        font-size: 1.1rem;
        color: #8888bb;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    /* ----- SIDEBAR ----- */
    .css-1d391kg, .css-1aumxhk {
        background: rgba(13, 13, 26, 0.92) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
    }
    .sidebar-brand {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sidebar-brand h1 {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f8cf7, #7c5cfc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
    }
    .sidebar-brand p {
        font-size: 0.75rem;
        color: #666699;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0;
    }
    .sidebar-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(79, 140, 247, 0.2), transparent);
        margin: 1rem 0;
    }
    .sidebar-status {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
    }
    .sidebar-status .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #666699;
    }
    .sidebar-status .value {
        font-size: 0.9rem;
        color: #d0d0e8;
        font-weight: 600;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.online { background: #2ecc71; box-shadow: 0 0 12px rgba(46, 204, 113, 0.4); }
    .status-dot.offline { background: #e74c3c; box-shadow: 0 0 12px rgba(231, 76, 60, 0.4); }
    
    /* ----- RADIO BUTTONS (sidebar nav) ----- */
    .stRadio > div {
        gap: 0.25rem;
    }
    .stRadio label {
        padding: 0.6rem 1rem !important;
        border-radius: 10px !important;
        transition: all 0.2s ease;
        font-weight: 500;
        color: #8888bb !important;
        background: transparent !important;
    }
    .stRadio label:hover {
        background: rgba(79, 140, 247, 0.08) !important;
        color: #d0d0e8 !important;
    }
    .stRadio label[data-selected="true"] {
        background: rgba(79, 140, 247, 0.12) !important;
        color: #4f8cf7 !important;
        border: 1px solid rgba(79, 140, 247, 0.15);
    }
    
    /* ----- BUTTONS ----- */
    .stButton button {
        background: linear-gradient(135deg, #4f8cf7, #7c5cfc) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(79, 140, 247, 0.25) !important;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 8px 32px rgba(79, 140, 247, 0.35) !important;
    }
    .stButton button:active {
        transform: scale(0.98);
    }
    
    /* ----- FORMS ----- */
    .stForm {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(8px);
    }
    
    /* ----- DATAFRAMES ----- */
    .dataframe-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.04);
        overflow: hidden;
    }
    .stDataFrame {
        border-radius: 8px !important;
    }
    
    /* ----- TABS ----- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        color: #8888bb !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(79, 140, 247, 0.12) !important;
        color: #4f8cf7 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #d0d0e8 !important;
    }
    
    /* ----- EXPANDER ----- */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #b8b8d0 !important;
    }
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 0 0 10px 10px !important;
        padding: 1rem !important;
    }
    
    /* ----- SELECTBOX, SLIDERS, INPUTS ----- */
    .stSelectbox, .stNumberInput, .stSlider {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #8888bb !important;
        font-weight: 500 !important;
    }
    input, select, .stSlider > div {
        color: #e8e8f0 !important;
    }
    
    /* ----- SUCCESS / WARNING / ERROR ----- */
    .stAlert {
        border-radius: 12px !important;
        backdrop-filter: blur(8px);
        border: none !important;
    }
    .stAlert[data-baseweb="notification"] {
        background: rgba(46, 204, 113, 0.08) !important;
        border-left: 3px solid #2ecc71 !important;
    }
    .stAlert[data-baseweb="notification"]:has(.stAlertWarning) {
        background: rgba(241, 196, 15, 0.08) !important;
        border-left: 3px solid #f1c40f !important;
    }
    .stAlert[data-baseweb="notification"]:has(.stAlertError) {
        background: rgba(231, 76, 60, 0.08) !important;
        border-left: 3px solid #e74c3c !important;
    }
    
    /* ----- BALLOONS (celebration) ----- */
    .stBalloons {
        z-index: 9999 !important;
    }
    
    /* ----- HIDE DEFAULT ELEMENTS ----- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }
    
    /* ----- SCROLLBAR ----- */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(79, 140, 247, 0.3);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(79, 140, 247, 0.5);
    }
</style>
"""
st.markdown(MAIN_CSS, unsafe_allow_html=True)

# =============================================================================
# MATPLOTLIB DARK THEME
# =============================================================================
plt.rcParams.update({
    "axes.facecolor": "#111128",
    "figure.facecolor": "#111128",
    "axes.edgecolor": "#2a2a5a",
    "axes.labelcolor": "#8888bb",
    "xtick.color": "#7878aa",
    "ytick.color": "#7878aa",
    "text.color": "#ccccee",
    "grid.color": "#1e1e3f",
    "grid.alpha": 0.4,
    "legend.facecolor": "#111128",
    "legend.edgecolor": "#2a2a5a",
})
sns.set_style("darkgrid")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def bento_card(content, key=None):
    """Render content inside a bento-style card."""
    st.markdown(f'<div class="bento-card" key="{key}">{content}</div>', unsafe_allow_html=True)

def render_metric(title, value, unit="", sub=""):
    """Render a metric inside a bento card with gradient text."""
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    html = f"""
    <div class="bento-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value-lg">{value} {unit_html}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_metric_row(metrics):
    """Render a row of metrics (list of dicts: title, value, unit, sub)."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_metric(**m)

def render_section_header(title, subtitle=""):
    """Render a section header with optional subtitle."""
    html = f"""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #e8e8f0, #8888bb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.25rem;">{title}</h1>
        {f'<p style="color: #666699; font-size: 1rem; margin-top: 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_subheader(title):
    st.markdown(f'<h3 style="color: #c8c8e0; font-weight: 600; margin: 1.5rem 0 0.75rem 0;">{title}</h3>', unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & PREPROCESSING (CACHED)
# =============================================================================
@st.cache_data
def load_data():
    """Load the dataset (fallback to synthetic data if file not found)."""
    try:
        df = pd.read_csv("olist_orders_dataset.csv")
    except FileNotFoundError:
        # Generate synthetic data for demo
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
    
    # Cap outliers
    df["delivery_days"] = df["delivery_days"].clip(lower=0, upper=60)
    df["estimated_days"] = df["estimated_days"].clip(lower=1, upper=60)
    
    return df

# Load and preprocess
with st.spinner("Loading data..."):
    raw_df = load_data()
    df = preprocess_data(raw_df)

# Feature definitions
FEATURES = ["purchase_hour", "purchase_day", "purchase_month", "purchase_weekday", "estimated_days"]
TARGET = "delivery_days"

# =============================================================================
# SESSION STATE INIT
# =============================================================================
for key in ["model", "scaler", "model_name", "trained", "metrics"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h1>📦</h1>
        <p>Delivery Time<br>Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 EDA", "⚙️ Preprocessing", "🧠 Train Model", "📈 Evaluation", "🎯 Predict"],
        label_visibility="collapsed",
    )
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    # Dataset status
    st.markdown("""
    <div class="sidebar-status">
        <div class="label">📊 Dataset</div>
        <div class="value">{:,} rows · {} features</div>
    </div>
    """.format(len(df), len(FEATURES)), unsafe_allow_html=True)
    
    # Model status
    if st.session_state.trained:
        status_color = "#2ecc71"
        status_text = "Online"
        model_name = st.session_state.model_name or "Model"
    else:
        status_color = "#e74c3c"
        status_text = "Offline"
        model_name = "Not trained"
    
    st.markdown(f"""
    <div class="sidebar-status">
        <div class="label">🧠 Model</div>
        <div class="value"><span class="status-dot {status_text.lower()}"></span>{model_name} · {status_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    
    st.caption("⚡ Built with Streamlit · v2.0")

# =============================================================================
# PAGE ROUTING
# =============================================================================

# ---- HOME ----
if page == "🏠 Home":
    render_section_header("📦 Delivery Time Prediction", "Predict delivery duration using machine learning")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("Total Orders", f"{len(df):,}")
    with col2:
        render_metric("Avg Delivery", f"{df[TARGET].mean():.1f}", "days")
    with col3:
        render_metric("Max Delivery", f"{int(df[TARGET].max())}", "days")
    with col4:
        render_metric("Std Dev", f"{df[TARGET].std():.1f}", "days")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_subheader("📋 Data Preview")
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(df.head(8), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        render_subheader("📈 Quick Stats")
        st.markdown(f"""
        <div class="bento-card">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <div><span style="color: #666699;">Min</span><br><span style="color: #e8e8f0; font-weight: 600;">{int(df[TARGET].min())} days</span></div>
                <div><span style="color: #666699;">Max</span><br><span style="color: #e8e8f0; font-weight: 600;">{int(df[TARGET].max())} days</span></div>
                <div><span style="color: #666699;">Median</span><br><span style="color: #e8e8f0; font-weight: 600;">{int(df[TARGET].median())} days</span></div>
                <div><span style="color: #666699;">Q1–Q3</span><br><span style="color: #e8e8f0; font-weight: 600;">{int(df[TARGET].quantile(0.25))} – {int(df[TARGET].quantile(0.75))} days</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---- EDA ----
elif page == "📊 EDA":
    render_section_header("🔍 Exploratory Data Analysis", "Understand your delivery data")
    
    # Quick metrics
    render_metric_row([
        {"title": "Total Orders", "value": f"{len(df):,}"},
        {"title": "Features", "value": len(FEATURES)},
        {"title": "Avg Delivery", "value": f"{df[TARGET].mean():.1f}", "unit": "days"},
        {"title": "Missing Values", "value": f"{df.isnull().sum().sum():,}"},
    ])
    
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📈 Correlations", "📋 Profile"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Delivery Days Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[TARGET], kde=True, color="#4f8cf7", ax=ax, bins=30)
            ax.set_xlabel("Delivery Days")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            feat = st.selectbox("Select feature", FEATURES)
            st.markdown(f"#### {feat} Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[feat], kde=True, color="#7c5cfc", ax=ax, bins=20)
            ax.set_xlabel(feat)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()
        
        st.markdown("#### Boxplots")
        cols = st.columns(3)
        for i, feat in enumerate(FEATURES[:3]):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.boxplot(y=df[feat], color="#4f8cf7", ax=ax)
                ax.set_ylabel(feat)
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        st.markdown("#### Feature Correlation Matrix")
        corr_df = df[FEATURES + [TARGET]].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, fmt=".2f", 
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        st.pyplot(fig)
        plt.close()
        
        st.markdown("#### Top Correlations with Target")
        corr_with_target = corr_df[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in corr_with_target.values]
        ax.barh(corr_with_target.index, corr_with_target.values, color=colors)
        ax.set_xlabel("Correlation with Delivery Days")
        ax.axvline(0, color="#666699", linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.markdown("#### Data Types")
        st.dataframe(df.dtypes.astype(str), use_container_width=True)
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

# ---- PREPROCESSING ----
elif page == "⚙️ Preprocessing":
    render_section_header("⚙️ Data Preprocessing", "Configure and run your preprocessing pipeline")
    
    with st.form("preprocessing_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05, format="%.2f")
        with col2:
            random_state = st.number_input("Random Seed", 1, 999, 42)
        with col3:
            scaler_option = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])
        
        submitted = st.form_submit_button("🚀 Run Preprocessing", use_container_width=True)
    
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
            
            # Save scaler
            with open("scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            
            st.success("✅ Preprocessing complete! Scaler saved as `scaler.pkl`")
            
            render_metric_row([
                {"title": "Training Set", "value": X_train_scaled.shape[0], "unit": "rows"},
                {"title": "Test Set", "value": X_test_scaled.shape[0], "unit": "rows"},
                {"title": "Features", "value": X_train_scaled.shape[1], "unit": "columns"},
                {"title": "Scaler", "value": scaler_option, "sub": "saved to disk"},
            ])
            
            # Show sample of scaled data
            with st.expander("🔍 Preview Scaled Data"):
                preview_df = pd.DataFrame(
                    X_train_scaled[:5],
                    columns=FEATURES
                )
                st.dataframe(preview_df, use_container_width=True)

# ---- TRAIN MODEL ----
elif page == "🧠 Train Model":
    render_section_header("🧠 Model Training", "Train a regression model on your preprocessed data")
    
    model_options = {
        "Linear Regression": LinearRegression(),
        "Random Forest (50 trees)": RandomForestRegressor(n_estimators=50, random_state=42),
        "Random Forest (100 trees)": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR (RBF)": SVR(kernel="rbf"),
        "SVR (Linear)": SVR(kernel="linear"),
    }
    
    selected_model = st.selectbox("Select Algorithm", list(model_options.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        train_btn = st.button("🚀 Train Model", use_container_width=True, type="primary")
    with col2:
        if st.button("🗑️ Clear Model", use_container_width=True):
            st.session_state.model = None
            st.session_state.scaler = None
            st.session_state.model_name = None
            st.session_state.trained = False
            st.session_state.metrics = None
            st.rerun()
    
    if train_btn:
        if not os.path.exists("scaler.pkl"):
            st.error("⚠️ Please run Preprocessing first to generate `scaler.pkl`")
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
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                metrics = {
                    "R²": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                }
                
                # Save
                with open("MainModel.pkl", "wb") as f:
                    pickle.dump(model, f)
                
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.model_name = selected_model
                st.session_state.trained = True
                st.session_state.metrics = metrics
                
                st.success(f"✅ {selected_model} trained successfully!")
                
                render_metric_row([
                    {"title": "R² Score", "value": f"{metrics['R²']:.4f}"},
                    {"title": "MAE", "value": f"{metrics['MAE']:.3f}", "unit": "days"},
                    {"title": "RMSE", "value": f"{metrics['RMSE']:.3f}", "unit": "days"},
                ])
                
                with st.expander("📊 Model Details"):
                    if hasattr(model, "get_params"):
                        st.json(model.get_params())
                    st.write("Model type:", type(model).__name__)

# ---- EVALUATION ----
elif page == "📈 Evaluation":
    render_section_header("📈 Model Evaluation", "Assess your model's performance")
    
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
        
        # Metrics
        metrics = {
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        
        render_metric_row([
            {"title": "R² Score", "value": f"{metrics['R²']:.4f}"},
            {"title": "MAE", "value": f"{metrics['MAE']:.3f}", "unit": "days"},
            {"title": "MSE", "value": f"{metrics['MSE']:.3f}", "unit": "days²"},
            {"title": "RMSE", "value": f"{metrics['RMSE']:.3f}", "unit": "days"},
        ])
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#4f8cf7", ax=ax)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, alpha=0.6)
            ax.set_xlabel("Actual Delivery Days")
            ax.set_ylabel("Predicted Delivery Days")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Residuals Distribution")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.histplot(residuals, kde=True, color="#7c5cfc", ax=ax, bins=25)
            ax.axvline(0, color="red", linestyle="--", alpha=0.6)
            ax.set_xlabel("Residual (Actual - Predicted)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()
        
        with st.expander("📋 Full Metrics Table"):
            st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}), use_container_width=True)
    
    else:
        st.warning("⚠️ No trained model found. Please go to **Train Model** and train a model first.")

# ---- PREDICT ----
else:  # "🎯 Predict"
    render_section_header("🎯 Make a Prediction", "Enter order details to estimate delivery time")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            purchase_hour = st.slider("Purchase Hour (0–23)", 0, 23, 12)
            purchase_day = st.number_input("Purchase Day (1–31)", 1, 31, 15)
            purchase_weekday = st.selectbox("Purchase Weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=2)
            weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
            purchase_weekday_val = weekday_map[purchase_weekday]
        
        with col2:
            purchase_month = st.selectbox("Purchase Month", list(range(1, 13)), index=5)
            estimated_days = st.number_input("Carrier Estimated Days", 1, 60, 10)
        
        submitted = st.form_submit_button("🔮 Predict Delivery Time", use_container_width=True, type="primary")
    
    if submitted:
        if not st.session_state.trained or st.session_state.model is None:
            st.error("⚠️ No trained model found. Please train a model first.")
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
            prediction = max(0, prediction)  # No negative days
            
            # Confidence interval approximation (using training residuals)
            # Simple: ±1.96 * RMSE if available
            if st.session_state.metrics and "RMSE" in st.session_state.metrics:
                rmse = st.session_state.metrics["RMSE"]
                ci_lower = max(0, prediction - 1.96 * rmse)
                ci_upper = prediction + 1.96 * rmse
            else:
                ci_lower = max(0, prediction - 3)
                ci_upper = prediction + 3
            
            # Display result
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">📦 Estimated Delivery Time</div>
                <div class="result-value">{prediction:.1f} <span style="font-size: 1.8rem; -webkit-text-fill-color: #8888bb;">days</span></div>
                <div style="margin-top: 0.75rem; color: #666699; font-size: 0.9rem;">
                    95% CI: {ci_lower:.1f} – {ci_upper:.1f} days
                </div>
                <div style="margin-top: 0.5rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                    <span style="background: rgba(79,140,247,0.1); padding: 0.25rem 1rem; border-radius: 20px; color: #8888bb; font-size: 0.8rem;">
                        🕐 {purchase_hour}:00
                    </span>
                    <span style="background: rgba(79,140,247,0.1); padding: 0.25rem 1rem; border-radius: 20px; color: #8888bb; font-size: 0.8rem;">
                        📅 {purchase_day}/{purchase_month}
                    </span>
                    <span style="background: rgba(79,140,247,0.1); padding: 0.25rem 1rem; border-radius: 20px; color: #8888bb; font-size: 0.8rem;">
                        🚚 {estimated_days} days estimated
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                st.json({
                    "purchase_hour": purchase_hour,
                    "purchase_day": purchase_day,
                    "purchase_month": purchase_month,
                    "purchase_weekday": purchase_weekday,
                    "estimated_days": estimated_days,
                    "prediction": round(prediction, 2),
                })

# =============================================================================
# FOOTER (visible on all pages)
# =============================================================================
st.markdown("""
<div style="text-align: center; padding: 2rem 0 0.5rem 0; border-top: 1px solid rgba(255,255,255,0.03); margin-top: 2rem;">
    <span style="color: #444466; font-size: 0.75rem;">
        📦 Delivery Time Predictor · Built with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
