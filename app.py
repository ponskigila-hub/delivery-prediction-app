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
# PAGE CONFIG & CSS
# =====================================
st.set_page_config(
    page_title='Delivery Time Prediction',
    page_icon='📦',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Responsive CSS that works in both Light and Dark mode
st.markdown("""
<style>
    /* Styled Metric Cards */
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 15px 20px;
        background: linear-gradient(90deg, rgba(28, 131, 225, 0.1) 0%, transparent 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    
    /* Prominent Prediction Output */
    .prediction-box {
        text-align: center;
        padding: 2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(28, 131, 225, 0.2), rgba(28, 131, 225, 0.05));
        border: 1px solid rgba(28, 131, 225, 0.3);
        margin-top: 1rem;
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1C83E1; /* Streamlit Primary Blue */
        margin: 0;
    }
    
    /* Clean up default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD & PREPROCESS DATA
# =====================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv('olist_orders_dataset.csv')
    except FileNotFoundError:
        # Fallback for demo purposes if CSV is missing
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
        [
            'Home',
            'Exploratory Data Analysis',
            'Data Preprocessing',
            'Train Your Model',
            'Model Evaluation',
            'Prediction Demo'
        ]
    )

# =====================================
# HOME
# =====================================
if menu == 'Home':
    st.title('Delivery Time Prediction App')
    st.markdown("Predict delivery duration based on order information and historical logistics data.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Injecting Custom HTML/CSS specifically for the Metric Cards
    st.markdown("""
    <style>
    .custom-metric-card {
        background: linear-gradient(135deg, rgba(28, 131, 225, 0.15) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(28, 131, 225, 0.3);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .custom-metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(28, 131, 225, 0.6);
    }
    .metric-title {
        font-size: 1rem;
        color: #A0AEC0; /* Light gray for dark mode readability */
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4299E1; /* Bright blue */
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render Custom Metric Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="custom-metric-card">
            <div class="metric-title">Total Orders</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="custom-metric-card">
            <div class="metric-title">Average Delivery</div>
            <div class="metric-value">{round(df[target].mean(), 1)} Days</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="custom-metric-card">
            <div class="metric-title">Max Delivery Time</div>
            <div class="metric-value">{int(df[target].max())} Days</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# =====================================
# EDA
# =====================================
elif menu == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')
    
    tab1, tab2, tab3 = st.tabs(["Data Profiling", "Feature Distributions", "Correlations"])
    
    with tab1:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Data Types")
            st.dataframe(df.dtypes.astype(str), use_container_width=True)
        with colB:
            st.subheader("Missing Values")
            st.dataframe(df.isnull().sum(), use_container_width=True)
            
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        selected_feature = st.selectbox('Select Feature to Analyze', features)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Histogram: {selected_feature}**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[selected_feature], kde=True, ax=ax, color='#1C83E1')
            st.pyplot(fig)
            
        with col2:
            st.markdown(f"**Boxplot: {selected_feature}**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[selected_feature], ax=ax, color='#1C83E1')
            st.pyplot(fig)

    with tab3:
        st.markdown("**Feature Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap='Blues', ax=ax, fmt=".2f")
        st.pyplot(fig)

# =====================================
# PREPROCESSING
# =====================================
elif menu == 'Data Preprocessing':
    st.title('Data Preprocessing')
    st.markdown("Configure hyperparameters for your data pipeline.")

    with st.form("preprocessing_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider('Test Size %', 0.1, 0.5, 0.2, 0.05)
        with col2:
            random_state = st.number_input('Random State', 1, 100, 42)
        with col3:
            scaler_option = st.selectbox('Scaler', ['StandardScaler', 'MinMaxScaler'])
            
        submit = st.form_submit_button("Run Preprocessing Pipeline", use_container_width=True)

    if submit:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        scaler = StandardScaler() if scaler_option == 'StandardScaler' else MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        st.success('✅ Pipeline executed successfully!')
        colA, colB = st.columns(2)
        colA.info(f"**Training Set:** {X_train_scaled.shape[0]} rows")
        colB.info(f"**Testing Set:** {X_test_scaled.shape[0]} rows")

# =====================================
# TRAIN MODEL
# =====================================
elif menu == 'Train Your Model':
    st.title('Train Your Model')

    model_option = st.selectbox(
        'Select Algorithm',
        ['Linear Regression (Baseline)', 'Random Forest Regressor (Proposed)', 'SVR (Alternative)']
    )

    if st.button('Initialize & Train Model', type="primary", use_container_width=True):
        with st.spinner('Training in progress...'):
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            if 'Linear Regression' in model_option:
                model = LinearRegression()
            elif 'Random Forest' in model_option:
                model = RandomForestRegressor(n_estimators=50) # Reduced for demo speed
            else:
                model = SVR()

            model.fit(X_train, y_train)

            with open('MainModel.pkl', 'wb') as file:
                pickle.dump(model, file)

            st.success(f'✅ {model_option.split("(")[0].strip()} trained and saved to disk!')

# =====================================
# MODEL EVALUATION
# =====================================
elif menu == 'Model Evaluation':
    st.title('Model Evaluation')

    if os.path.exists('MainModel.pkl') and os.path.exists('scaler.pkl'):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('MainModel.pkl', 'rb') as file:
            model = pickle.load(file)

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('R² Score', round(r2_score(y_test, y_pred), 3))
        col2.metric('MAE', round(mean_absolute_error(y_test, y_pred), 2))
        col3.metric('MSE', round(mean_squared_error(y_test, y_pred), 2))
        col4.metric('RMSE', round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
        
        st.markdown("### Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='#1C83E1', ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Delivery Days")
        ax.set_ylabel("Predicted Delivery Days")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please run the Preprocessing and Training steps first.")

# =====================================
# PREDICTION DEMO
# =====================================
elif menu == 'Prediction Demo':
    st.title('Prediction Demo')
    st.markdown("Input new order parameters to predict the delivery duration.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            purchase_hour = st.slider('Purchase Hour (0-23)', 0, 23, 12)
            purchase_day = st.number_input('Purchase Day (1-31)', 1, 31, 15)
            
        with col2:
            purchase_month = st.selectbox('Purchase Month', list(range(1, 13)), index=5)
            estimated_days = st.number_input('Estimated Delivery Days', 1, 60, 10)

        submit_prediction = st.form_submit_button('Predict Delivery Time', type="primary", use_container_width=True)

    if submit_prediction:
        if os.path.exists('MainModel.pkl') and os.path.exists('scaler.pkl'):
            with open('MainModel.pkl', 'rb') as file:
                model = pickle.load(file)
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)

            input_data = np.array([[purchase_hour, purchase_day, purchase_month, estimated_days]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            st.markdown(f"""
                <div class="prediction-box">
                    <h3>Estimated Transit Time</h3>
                    <p class="prediction-value">{round(prediction, 1)} <span style="font-size: 1.5rem; color: #555;">Days</span></p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.error("⚠️ Predictive system offline. Please train a model first.")
