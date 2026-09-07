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
# PAGE CONFIG & CUSTOM CSS
# =====================================
st.set_page_config(
    page_title='United Logistics | Delivery Prediction',
    page_icon='🚛',
    layout='wide',
    initial_sidebar_state='expanded'
)

def inject_custom_css():
    st.markdown("""
    <style>
        /* Corporate Logistics Theme */
        :root {
            --primary-blue: #003366;
            --accent-orange: #FF6200;
            --bg-light: #F4F6F9;
        }
        
        /* Headers and text */
        h1, h2, h3 {
            color: var(--primary-blue) !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Primary Buttons (Simulating Call-to-Action) */
        .stButton>button {
            background-color: var(--accent-orange);
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #e55800;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Metric Cards */
        div[data-testid="stMetricValue"] {
            color: var(--primary-blue);
            font-size: 2.5rem !important;
            font-weight: 700;
        }
        div[data-testid="metric-container"] {
            background-color: white;
            border-top: 4px solid var(--accent-orange);
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--primary-blue);
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# =====================================
# LOAD & PREPROCESS DATA
# =====================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv('olist_orders_dataset.csv')
    except FileNotFoundError:
        # Fallback dummy data if file is missing so the UI still renders
        dates = pd.date_range(start='2023-01-01', periods=100)
        return pd.DataFrame({
            'order_purchase_timestamp': dates,
            'order_delivered_customer_date': dates + pd.to_timedelta(np.random.randint(2, 15, 100), unit='d'),
            'order_estimated_delivery_date': dates + pd.to_timedelta(np.random.randint(5, 20, 100), unit='d')
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
# SIDEBAR NAVIGATION
# =====================================
with st.sidebar:
    st.markdown("## 🚛 United Logistics")
    st.markdown("---")
    menu = st.radio(
        "NETWORK OPERATIONS",
        [
            '🏠 Dashboard',
            '📊 Market Analysis',
            '⚙️ Pipeline Config',
            '🤖 Model Deployment',
            '📈 Diagnostics',
            '🚀 Get Delivery Quote'
        ]
    )
    st.markdown("---")
    st.caption("System Status: **Online** 🟢")

# =====================================
# ROUTING & VIEWS
# =====================================

if menu == '🏠 Dashboard':
    st.title('Global Fulfillment Dashboard')
    st.markdown("Monitor network efficiency and delivery lifecycles in real-time.")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Shipments Processed', f"{len(df):,}")
    col2.metric('Avg. Transit Time (Days)', round(df[target].mean(), 1))
    col3.metric('Max Transit Time (Days)', int(df[target].max()))
    
    st.markdown("### Recent Network Activity")
    st.dataframe(df.head(15), use_container_width=True)

elif menu == '📊 Market Analysis':
    st.title('Exploratory Data Analysis')
    st.markdown("Investigate operational bottlenecks and delivery trends.")
    
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Distributions", "Correlations"])
    
    with tab1:
        st.subheader("Data Types & Missing Values")
        colA, colB = st.columns(2)
        with colA:
            st.write(df.dtypes)
        with colB:
            st.write(df.isnull().sum())
            
    with tab2:
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox('Analyze Feature', features, key='hist_feat')
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_feature], kde=True, color='#003366', ax=ax)
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Network Feature Correlation")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap='Oranges', ax=ax)
        st.pyplot(fig)

elif menu == '⚙️ Pipeline Config':
    st.title('Data Preprocessing')
    st.markdown("Configure the data pipeline before model training.")
    
    with st.form("preprocessing_form"):
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider('Holdout Set Size (Test %)', 0.1, 0.5, 0.2)
        with col2:
            scaler_option = st.selectbox('Scaling Algorithm', ['StandardScaler', 'MinMaxScaler'])
            
        random_state = st.number_input('Random Seed', value=42)
        submit_prep = st.form_submit_button("Initialize Pipeline")
        
    if submit_prep:
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        scaler = StandardScaler() if scaler_option == 'StandardScaler' else MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)
            
        st.success('✅ Pipeline initialized and scaler serialized successfully.')
        colA, colB = st.columns(2)
        colA.info(f"Training Matrix: {X_train_scaled.shape}")
        colB.info(f"Target Vector: {y_train.shape}")

elif menu == '🤖 Model Deployment':
    st.title('Model Training Hub')
    st.markdown("Select and deploy machine learning architectures for ETA prediction.")
    
    model_option = st.selectbox(
        'Select Architecture',
        ['Linear Regression (Baseline)', 'Random Forest Regressor (Proposed)', 'SVR (Alternative)']
    )
    
    if st.button('Deploy Model', use_container_width=True):
        with st.spinner('Compiling and training model...'):
            X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            
            if 'Linear Regression' in model_option:
                model = LinearRegression()
            elif 'Random Forest' in model_option:
                model = RandomForestRegressor(n_estimators=50) # limited for speed in demo
            else:
                model = SVR()
                
            model.fit(X_train, y_train)
            
            with open('MainModel.pkl', 'wb') as file:
                pickle.dump(model, file)
                
            st.success(f'✅ {model_option.split(" ")[0]} deployed successfully to production!')

elif menu == '📈 Diagnostics':
    st.title('Performance Diagnostics')
    
    if os.path.exists('MainModel.pkl') and os.path.exists('scaler.pkl'):
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('MainModel.pkl', 'rb') as file:
            model = pickle.load(file)
            
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", round(r2_score(y_test, y_pred), 3))
        col2.metric("MAE", round(mean_absolute_error(y_test, y_pred), 2))
        col3.metric("MSE", round(mean_squared_error(y_test, y_pred), 2))
        col4.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
        
        # Actual vs Predicted Plot
        st.markdown("### Actual vs. Predicted Delivery Days")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='#003366', ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Days")
        ax.set_ylabel("Predicted Days")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No compiled model found. Please run the Pipeline and Training sequences first.")

elif menu == '🚀 Get Delivery Quote':
    st.title('Delivery Time Estimator')
    st.markdown("Generate AI-powered delivery estimates for your logistics planning.")
    
    # Styled like a "Tracking / Quote" widget on a logistics site
    with st.container():
        st.markdown("""
            <div style='background-color: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #FF6200;'>
                <h3 style='margin-top: 0; color: #003366;'>Calculate Estimated Time of Arrival (ETA)</h3>
            </div>
            <br>
        """, unsafe_allow_html=True)
        
        with st.form("quote_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                purchase_hour = st.slider('Time of Order (Hour)', 0, 23, 12, help="24-hour format")
                purchase_day = st.number_input('Day of Month', 1, 31, 15)
                
            with col2:
                purchase_month = st.selectbox('Month of Order', list(range(1, 13)), index=5)
                estimated_days = st.number_input('Carrier Estimated Days', 1, 60, 10, help="Initial carrier estimate")

            submit_quote = st.form_submit_button("Calculate Accurate ETA", use_container_width=True)

    if submit_quote:
        if os.path.exists('MainModel.pkl') and os.path.exists('scaler.pkl'):
            with open('MainModel.pkl', 'rb') as file:
                model = pickle.load(file)
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)

            input_data = np.array([[purchase_hour, purchase_day, purchase_month, estimated_days]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            st.markdown(f"""
                <div style='text-align: center; background-color: #e6f3ff; padding: 2rem; border-radius: 8px; margin-top: 2rem; border: 1px solid #b3d9ff;'>
                    <h2 style='color: #003366; margin: 0;'>Predicted Transit Time</h2>
                    <h1 style='color: #FF6200; font-size: 4rem; margin: 0;'>{round(prediction, 1)} <span style='font-size: 2rem; color: #003366;'>Days</span></h1>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.error("⚠️ Predictive system offline. Please configure pipeline and train models first.")
