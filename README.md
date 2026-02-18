# 🚆 Metro Ridership Forecasting & Capacity Planning

## 📌 Project Overview
This project develops a **data-driven forecasting model** to support metro **capacity planning and train allocation decisions** during and after the COVID-19 period.

Using a real-world case study inspired by the Toronto metro system (Canada), the objective is to predict passenger demand across stations and time periods and translate predictions into operational train allocation decisions.

Unlike station-specific modeling approaches, this project implements a **single global ElasticNet regression model** trained across all stations to ensure scalability, robustness, and generalization.

---

## 🎯 Objectives

- Analyze metro ridership patterns before and during COVID-19  
- Engineer time-series features capturing temporal demand dynamics  
- Train a regularized regression model to predict passenger volume  
- Convert predicted demand into train allocation decisions using capacity constraints  
- Evaluate performance using robust statistical metrics  

---

## 📂 Dataset Description

The dataset contains operational and temporal features:

- **Year, Month, Day, Week Number**
- **Station**
- **Corridor / Route intersections**
- **Working day indicator**
- **Time period of the day** (Morning, Noon, Evening, Night)
- **Number of passengers (Ridership)** — Target variable
- **COVID-19 alert status**

> ⚠️ The variable *Number of trains* was excluded from modeling to prevent target leakage and circular decision-making.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Time-aware train/test split based on chronological order  
- Handling missing and infinite values  
- One-Hot Encoding for categorical features  
- Feature scaling using `StandardScaler`  

### 2️⃣ Feature Engineering
- Lag features (previous ridership values)  
- Rolling mean features  
- Cyclical encoding for seasonal effects (month)  
- COVID impact indicator  

### 3️⃣ Model Development

A single **ElasticNet regression model** was trained across all stations.

ElasticNet was selected because it:

- Handles multicollinearity  
- Performs automatic feature selection  
- Reduces overfitting through regularization  
- Is robust to high-dimensional encoded features  

Hyperparameter tuning was performed using **TimeSeriesSplit cross-validation** to prevent data leakage.

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- **Root Mean Squared Error (RMSE)**

- **Coefficient of Determination (R²)**

Evaluation is conducted on a strictly held-out time-based test set.

---

## 📁 Project Structure
├── data/                # Raw dataset
├── notebooks/           # EDA & Modeling notebooks
├── Model.pkl            # Trained ElasticNet model
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

---

## 🏆 Key Outcomes

- A scalable demand forecasting model applicable across all stations

- Quantified impact of COVID-19 on ridership

- A transparent framework for translating forecasts into train allocation

- A fully reproducible ML pipeline with time-aware validation

---

## 🔮 Future Improvements

- Incorporating external variables (weather, events, holidays)

- Fleet-level optimization under operational constraints

- Comparison with tree-based and deep learning models

- Multi-step forecasting framework