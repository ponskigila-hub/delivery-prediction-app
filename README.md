# 📦 Delivery Time Prediction App

A machine learning web application built with <a href="https://streamlit.io/">Streamlit</a> to predict customer delivery time based on order information from the Olist e-commerce dataset.

---

## 🚀 Features

- 📊 Interactive Exploratory Data Analysis (EDA)
- 🧹 Data preprocessing and feature engineering
- 🤖 Train multiple machine learning models
- 📈 Model evaluation with regression metrics
- 🔮 Real-time delivery time prediction
- 💾 Save trained models and scalers using Pickle

---

## 🧠 Machine Learning Models

This project supports multiple regression algorithms:

- Linear Regression (Baseline)
- Random Forest Regressor (Proposed Model)
- Support Vector Regressor (SVR)

---

## 📂 Dataset

Dataset used:

- `olist_orders_dataset.csv`

The dataset should contain these columns:

- `order_purchase_timestamp`
- `order_delivered_customer_date`
- `order_estimated_delivery_date`

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/delivery-time-prediction.git
cd delivery-time-prediction
```

### 2. Create virtual environment (optional)

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📦 Required Libraries

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
pickle-mixin
```

---

## 🔍 Application Pages

### 🏠 Home
- Displays dataset overview
- Shows delivery statistics
- Preview dataset

### 📊 Exploratory Data Analysis
- Histogram visualization
- Correlation heatmap
- Scatterplot analysis
- Boxplot visualization

### 🧹 Data Preprocessing
- Train-test split
- Feature scaling
- Save scaler using Pickle

### 🤖 Train Your Model
- Select regression model
- Train and save model

### 📈 Model Evaluation
Evaluation metrics:
- R² Score
- MAE
- MSE
- RMSE

### 🔮 Prediction Demo
Predict delivery duration using:
- Purchase hour
- Purchase day
- Purchase month
- Estimated delivery days

---

## 🛠 Feature Engineering

Generated features:

| Feature | Description |
|---|---|
| purchase_hour | Hour of purchase |
| purchase_day | Day of purchase |
| purchase_month | Month of purchase |
| estimated_days | Estimated shipping duration |

Target variable:

| Target | Description |
|---|---|
| delivery_days | Actual delivery duration |

---

## 📁 Project Structure

```bash
delivery-time-prediction/
│
├── app.py
├── olist_orders_dataset.csv
├── MainModel.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 📸 Sample Workflow

1. Load dataset
2. Perform preprocessing
3. Train regression model
4. Evaluate model performance
5. Predict delivery time

---

## 📊 Evaluation Metrics Formula

- **MAE** → Mean Absolute Error
- **MSE** → Mean Squared Error
- **RMSE** → Root Mean Squared Error
- **R² Score** → Coefficient of Determination

---

## 💡 Future Improvements

- Hyperparameter tuning
- Add more delivery-related features
- Deploy to Streamlit Cloud
- Add model comparison dashboard
- Improve prediction accuracy

---

## 👨‍💻 Author

Developed for machine learning and data science learning purposes using Streamlit and Scikit-learn.

---

## 📜 License

This project is open-source and available under the MIT License.
