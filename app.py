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

# =====================================
# PAGE CONFIG & GLOBAL CSS
# =====================================
st.set_page_config(
    page_title='Delivery Time Prediction',
    page_icon='📦',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    /* ----- Global ----- */
    .main > div {
        padding-top: 2rem;
    }
    .block-container {
        padding-bottom: 3rem;
    }

    /* ----- Metric Cards ----- */
    .custom-metric-card {
        background: linear-gradient(135deg, rgba(28, 131, 225, 0.12) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(28, 131, 225, 0.3);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 1rem;
        backdrop-filter: blur(4px);
    }
    .custom-metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(28, 131, 225, 0.6);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #A0AEC0;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #4299E1;
        margin: 0;
    }

    /* ----- Prediction Card ----- */
    .prediction-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15) 0%, rgba(39, 174, 96, 0.05) 100%);
        border: 1px solid rgba(46, 204, 113, 0.4);
        padding: 3rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    .prediction-title {
        color: #A0AEC0;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    .prediction-value {
        font-size: 4.5rem;
        font-weight: 900;
        color: #2ECC71;
        margin: 0;
        line-height: 1;
    }

    /* ----- Section Headers ----- */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #EDF2F7;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid rgba(28, 131, 225, 0.3);
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #CBD5E0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ----- Form & Buttons ----- */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(28, 131, 225, 0.4);
    }
    .stForm {
        background: rgba(255,255,255,0.03);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* ----- Dataframe ----- */
    .dataframe-container {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================
# UI HELPER FUNCTIONS
# =====================================
def render_card(title, value, unit="", color_override=None, border_color=None):
    """Render a custom metric card with optional colour overrides."""
    color_style = f"color: {color_override};" if color_override else ""
    border_style = ""
    if border_color:
        border_style = f"border-color: {border_color};"
    if color_override:
        # also adjust background gradient based on colour
        bg_grad = f"background: linear-gradient(135deg, {color_override}15 0%, transparent 100%);"
        border_style += bg_grad

    html = f"""
    <div class="custom-metric-card" style="{border_style}">
        <div class="metric-title" style="{color_style}">{title}</div>
        <div class="metric-value" style="{color_style}">{value} <span style="font-size: 1rem; color: #A0AEC0;">{unit}</span></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def render_subsection_header(text):
    st.markdown(f'<div class="subsection-header">{text}</div>', unsafe_allow_html=True)

# =====================================
# LOAD & PREPROCESS DATA
# =====================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv('olist_orders_dataset.csv')
    except FileNotFoundError:
        dates = pd.date_range(start='2023-01-01', periods=200)
        return pd.DataFrame({
            'order_purchase_timestamp': dates,
            'order_delivered_customer_date': dates + pd.to_timedelta(np.random.randint(2, 15, 200), unit='d'),
            'order_estimated_delivery_date': dates + pd.to_timedelta(np.random.randint(5, 20, 200), unit='d')
        })

@st.cache_data
def preprocess(df):
    df = df.copy()
    df = df.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])
    
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_day'] = df['order_purchase_timestamp'].dt.day
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month

    df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['estimated_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
    return df

df = load_data()
df = preprocess(df)

features = ['purchase_hour', 'purchase_day', 'purchase_month', 'estimated_days']
target = 'delivery_days'

# =====================================
# SIDEBAR MENU
# =====================================
with st.sidebar:
    st.title('📦 Navigation')
    st.markdown("---")
    menu = st.radio(
        'Go to:',
        ['Home', 'Exploratory Data Analysis', 'Data Preprocessing', 'Train Your Model', 'Model Evaluation', 'Prediction Demo']
    )

# =====================================
# ROUTING
# =====================================
if menu == 'Home':
    render_section_header('📦 Delivery Time Prediction App')
    st.markdown("Predict delivery duration based on order information and historical logistics data.")

    col1, col2, col3 = st.columns(3)
    with col1: render_card("Total Orders", f"{len(df):,}")
    with col2: render_card("Average Delivery", round(df[target].mean(), 1), "Days")
    with col3: render_card("Max Delivery Time", int(df[target].max()), "Days")

    render_subsection_header("📋 Dataset Preview")
    with st.container():
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif menu == 'Exploratory Data Analysis':
    render_section_header('🔍 Exploratory Data Analysis')
    tab1, tab2, tab3 = st.tabs(["Data Profiling", "Feature Distributions", "Correlations"])
    
    with tab1:
        render_subsection_header("Data Overview")
        colA, colB, colC = st.columns(3)
        with colA: render_card("Rows", f"{df.shape[0]:,}")
        with colB: render_card("Columns", f"{df.shape[1]}")
        with colC: render_card("Missing Values", f"{df.isnull().sum().sum():,}")

        col1, col2 = st.columns(2)
        with col1:
            render_subsection_header("Data Types")
            st.dataframe(df.dtypes.astype(str), use_container_width=True)
        with col2:
            render_subsection_header("Missing Values per Column")
            st.dataframe(df.isnull().sum(), use_container_width=True)

        render_subsection_header("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        selected_feature = st.selectbox('Select Feature to Analyze', features)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Histogram: {selected_feature}**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[selected_feature], kde=True, ax=ax, color='#1C83E1')
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            st.pyplot(fig)
        with col2:
            st.markdown(f"**Boxplot: {selected_feature}**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[selected_feature], ax=ax, color='#1C83E1')
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            st.pyplot(fig)

    with tab3:
        render_subsection_header("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap='Blues', ax=ax, fmt=".2f")
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        st.pyplot(fig)

elif menu == 'Data Preprocessing':
    render_section_header('⚙️ Data Preprocessing')
    st.markdown("Configure hyperparameters for your data pipeline.")

    with st.form("preprocessing_form"):
        col1, col2, col3 = st.columns(3)
        with col1: test_size = st.slider('Test Size %', 0.1, 0.5, 0.2, 0.05)
        with col2: random_state = st.number_input('Random State', 1, 100, 42)
        with col3: scaler_option = st.selectbox('Scaler', ['StandardScaler', 'MinMaxScaler'])
        submit = st.form_submit_button("Run Preprocessing Pipeline", use_container_width=True)

    if submit:
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        scaler = StandardScaler() if scaler_option == 'StandardScaler' else MinMaxScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        with open('scaler.pkl', 'wb') as file: pickle.dump(scaler, file)

        st.success('✅ Pipeline executed successfully!')
        colA, colB, colC = st.columns(3)
        with colA: render_card("Training Set Size", X_train_scaled.shape[0], "Rows", border_color="rgba(46, 204, 113, 0.4)")
        with colB: render_card("Test Set Size", X_test_scaled.shape[0], "Rows", border_color="rgba(46, 204, 113, 0.4)")
        with colC: render_card("Features", X_train_scaled.shape[1], "Columns", border_color="rgba(46, 204, 113, 0.4)")
        st.markdown("---")
        st.markdown("**Scaler saved:** `scaler.pkl` ready for model training.")

elif menu == 'Train Your Model':
    render_section_header('🧠 Train Your Model')
    model_option = st.selectbox('Select Algorithm', ['Linear Regression (Baseline)', 'Random Forest Regressor (Proposed)', 'SVR (Alternative)'])

    if st.button('Initialize & Train Model', type="primary", use_container_width=True):
        with st.spinner('Training in progress...'):
            X, y = df[features], df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            if 'Linear Regression' in model_option:
                model = LinearRegression()
                model_name = "Linear Regression"
            elif 'Random Forest' in model_option:
                model = RandomForestRegressor(n_estimators=50)
                model_name = "Random Forest"
            else:
                model = SVR()
                model_name = "SVR"

            model.fit(X_train, y_train)
            with open('MainModel.pkl', 'wb') as file: pickle.dump(model, file)
            
            st.success(f"✅ {model_name} trained and saved as `MainModel.pkl`")
            render_card("Status", "🟢 Online", f"{model_name} Deployed", color_override="#2ECC71", border_color="rgba(46, 204, 113, 0.4)")

            # Show model parameters summary
            with st.expander("Model Details"):
                if hasattr(model, 'get_params'):
                    st.json(model.get_params())
                else:
                    st.write("Model object:", model)

elif menu == 'Model Evaluation':
    render_section_header('📊 Model Evaluation')

    if os.path.exists('MainModel.pkl') and os.path.exists('scaler.pkl'):
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with open('scaler.pkl', 'rb') as file: scaler = pickle.load(file)
        with open('MainModel.pkl', 'rb') as file: model = pickle.load(file)

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        render_subsection_header("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1: render_card("R² Score", round(r2_score(y_test, y_pred), 3), border_color="rgba(28, 131, 225, 0.4)")
        with col2: render_card("MAE", round(mean_absolute_error(y_test, y_pred), 2), border_color="rgba(28, 131, 225, 0.4)")
        with col3: render_card("MSE", round(mean_squared_error(y_test, y_pred), 2), border_color="rgba(28, 131, 225, 0.4)")
        with col4: render_card("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2), border_color="rgba(28, 131, 225, 0.4)")

        render_subsection_header("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='#1C83E1', ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Delivery Days")
        ax.set_ylabel("Predicted Delivery Days")
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please run the Preprocessing and Training steps first.")

elif menu == 'Prediction Demo':
    render_section_header('🎯 Prediction Demo')
    st.markdown("Input new order parameters to predict the delivery duration.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            purchase_hour = st.slider('Purchase Hour (0-23)', 0, 23, 12)
            purchase_day = st.number_input('Purchase Day (1-31)', 1, 31, 15)
        with col2:
            purchase_month = st.selectbox('Purchase Month', list(range(1, 13)), index=5)
            estimated_days = st.number_input('Carrier Estimated Days', 1, 60, 10)
        submit_prediction = st.form_submit_button('Generate AI Prediction', type="primary", use_container_width=True)

    if submit_prediction:
        if os.path.exists('MainModel.pkl') and os.path.exists('scaler.pkl'):
            with open('MainModel.pkl', 'rb') as file: model = pickle.load(file)
            with open('scaler.pkl', 'rb') as file: scaler = pickle.load(file)

            input_data = np.array([[purchase_hour, purchase_day, purchase_month, estimated_days]])
            prediction = model.predict(scaler.transform(input_data))[0]

            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-title">Estimated Transit Time</div>
                    <p class="prediction-value">{round(prediction, 1)} <span style="font-size: 1.5rem; color: #A0AEC0;">Days</span></p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.error("⚠️ Predictive system offline. Please train a model first.")
