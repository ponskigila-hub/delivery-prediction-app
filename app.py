import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title='Delivery Time Prediction',
    page_icon='📦',
    layout='wide'
)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv('olist_orders_dataset.csv')

# =====================================
# PREPROCESSING FUNCTION
# =====================================
def preprocess(df):
    df = df.copy()

    # Drop rows with missing delivery dates
    df = df.dropna(subset=[
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ])

    # Convert datetime columns
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

    # Feature engineering
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_day'] = df['order_purchase_timestamp'].dt.day
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month

    # Target variable (actual delivery duration)
    df['delivery_days'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days

    # Estimated delivery duration
    df['estimated_days'] = (
        df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
    ).dt.days

    return df

# =====================================
# LOAD + PROCESS DATA
# =====================================
df = load_data()
df = preprocess(df)

features = [
    'purchase_hour',
    'purchase_day',
    'purchase_month',
    'estimated_days'
]

target = 'delivery_days'

# =====================================
# SIDEBAR MENU
# =====================================
st.sidebar.title('Navigation')
menu = st.sidebar.selectbox(
    'Choose Page',
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
    st.title('📦 Delivery Time Prediction App')
    st.write('Predict delivery duration based on order information.')

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Orders', len(df))
    col2.metric('Average Delivery Days', round(df[target].mean(), 2))
    col3.metric('Maximum Delivery Days', int(df[target].max()))

    st.subheader('Dataset Preview')
    st.dataframe(df.head())

# =====================================
# EDA
# =====================================
elif menu == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')

    st.subheader('Dataset Head')
    st.dataframe(df.head())

    st.subheader('Data Types')
    st.write(df.dtypes)

    st.subheader('Null Values')
    if st.checkbox('Show Null Values'):
        st.write(df.isnull().sum())

    st.subheader('Statistical Summary')
    st.write(df.describe())

    chart_option = st.selectbox(
        'Select Visualization',
        ['Histogram', 'Heatmap', 'Scatterplot', 'Boxplot']
    )

    selected_feature = st.selectbox('Select Feature', features)

    if chart_option == 'Histogram':
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        st.pyplot(fig)

    elif chart_option == 'Heatmap':
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif chart_option == 'Scatterplot':
        fig, ax = plt.subplots()
        sns.regplot(x=df[selected_feature], y=df[target], ax=ax)
        st.pyplot(fig)

    elif chart_option == 'Boxplot':
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_feature], ax=ax)
        st.pyplot(fig)

# =====================================
# PREPROCESSING
# =====================================
elif menu == 'Data Preprocessing':
    st.title('Data Preprocessing')

    test_size = st.slider('Test Size', 0.1, 0.5, 0.2)
    random_state = st.slider('Random State', 1, 100, 42)

    scaler_option = st.selectbox(
        'Select Scaler',
        ['StandardScaler', 'MinMaxScaler']
    )

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    if scaler_option == 'StandardScaler':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    st.success('Preprocessing completed')
    st.write('Train shape:', X_train_scaled.shape)
    st.write('Test shape:', X_test_scaled.shape)

# =====================================
# TRAIN MODEL
# =====================================
elif menu == 'Train Your Model':
    st.title('Train Your Model')

    model_option = st.selectbox(
        'Select Model',
        [
            'Linear Regression (Baseline)',
            'Random Forest Regressor (Proposed)',
            'SVR (Alternative)'
        ]
    )

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if st.button('Train Model'):
        if 'Linear Regression' in model_option:
            model = LinearRegression()
        elif 'Random Forest' in model_option:
            model = RandomForestRegressor()
        else:
            model = SVR()

        model.fit(X_train, y_train)

        with open('MainModel.pkl', 'wb') as file:
            pickle.dump(model, file)

        st.success('Model trained and saved successfully!')

# =====================================
# MODEL EVALUATION
# =====================================
elif menu == 'Model Evaluation':
    st.title('Model Evaluation')

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with open('MainModel.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test)

    st.subheader('Evaluation Metrics')
    st.write('R2 Score:', r2_score(y_test, y_pred))
    st.write('MAE:', mean_absolute_error(y_test, y_pred))
    st.write('MSE:', mean_squared_error(y_test, y_pred))
    st.write('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# =====================================
# PREDICTION DEMO
# =====================================
elif menu == 'Prediction Demo':
    st.title('Prediction Demo')

    purchase_hour = st.number_input('Purchase Hour', 0, 23, 12)
    purchase_day = st.number_input('Purchase Day', 1, 31, 15)
    purchase_month = st.number_input('Purchase Month', 1, 12, 6)
    estimated_days = st.number_input('Estimated Delivery Days', 1, 60, 10)

    input_data = np.array([
        [purchase_hour, purchase_day, purchase_month, estimated_days]
    ])

    if st.button('Predict Delivery Time'):
        with open('MainModel.pkl', 'rb') as file:
            model = pickle.load(file)

        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.success(f'Predicted Delivery Time: {round(prediction[0], 2)} days')
