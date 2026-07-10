# 🚗 Car Price Prediction — Machine Learning Project

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=2196F3&center=true&vCenter=true&width=500&lines=Car+Price+Prediction+%F0%9F%9A%97;4+ML+Models+Compared;R%C2%B2+%3D+0.9995+Accuracy+%F0%9F%94%A5;Linear+Regression+Wins!;CodeAlpha+Data+Science+Internship" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Internship-CodeAlpha-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Track-Data%20Science-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Best%20Model-R²%3D0.9995-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-10%2C000%20Cars-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Models%20Compared-4-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square"/>
</p>

---

## 📋 Project Overview

A complete **end-to-end machine learning project** that predicts used car prices based on vehicle attributes. This project compares **4 regression algorithms**, builds a full preprocessing pipeline, saves the trained model as `.pkl`, and provides a ready-to-use prediction function for real-world deployment.

> 🏆 **Best Result:** Linear Regression achieved **R² = 0.9995** — outperforming XGBoost, Gradient Boosting, and Random Forest!

**Internship:** CodeAlpha — Data Science Track
**Author:** Faiz Ali | [GitHub](https://github.com/Faizalik1) | [LinkedIn](https://www.linkedin.com/in/faiz-ali-03365b217)

---

## 🎯 Business Problem

The used car market faces major pricing challenges:

| Problem | Impact |
|---------|--------|
| 🤔 Buyers don't know if they're overpaying | Poor customer experience |
| 🤔 Sellers don't know fair market value | Revenue loss |
| 🤔 Dealerships need automated pricing | Manual, slow, inaccurate |

**Solution:** An ML model that predicts car prices accurately from vehicle features — enabling **data-driven, automated pricing decisions.**

---

## 📊 Dataset

**Size:** 10,000 cars | 10 columns | Clean data (no nulls, no duplicates)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Brand` | String | Car manufacturer | Toyota, BMW |
| `Model` | String | Specific model | Corolla, 3 Series |
| `Year` | Integer | Manufacturing year | 2019 |
| `Engine_Size` | Float | Engine size in liters | 2.5 |
| `Fuel_Type` | String | Fuel category | Petrol, Diesel, Electric |
| `Transmission` | String | Gearbox type | Manual, Automatic |
| `Mileage` | Integer | Total km driven | 45,000 |
| `Doors` | Integer | Number of doors | 4 |
| `Owner_Count` | Integer | Previous owners | 1 |
| **`Price`** | **Float** | **🎯 Target variable** | **$25,000** |

---

## 🔬 Complete Pipeline

```
1️⃣  DATA LOADING         → Load CSV, inspect shape, types, statistics
2️⃣  EDA                  → Price distribution, correlations, outlier check
3️⃣  FEATURE ENGINEERING  → Car_Age derived from Year (dropped Model column)
4️⃣  PREPROCESSING        → StandardScaler + OneHotEncoder in Pipeline
5️⃣  MODEL TRAINING       → 4 algorithms trained and compared
6️⃣  EVALUATION           → MAE, RMSE, R², Cross-Validation (5-fold)
7️⃣  FEATURE IMPORTANCE   → Top predictors identified and ranked
8️⃣  PREDICTION FUNCTION  → predict_car_price() built for new cars
9️⃣  MODEL SAVING         → Exported as car_price_model.pkl (joblib)
```

---

## 🤖 Model Comparison Results

| Rank | 🏆 Model | MAE | RMSE | R² Score |
|------|---------|-----|------|----------|
| 🥇 1st | **Linear Regression** | **19.71** | **64.77** | **0.9995** |
| 🥈 2nd | XGBoost | 116.27 | 150.95 | 0.9975 |
| 🥉 3rd | Gradient Boosting | 169.55 | 216.66 | 0.9949 |
| 4️⃣ 4th | Random Forest | 246.75 | 318.03 | 0.9890 |

> 💡 **Why did Linear Regression WIN?**
> The relationship between car price and key features (car age + mileage) is **largely linear** in this dataset. This proves an important data science lesson:
> **"The best model is not always the most complex one!"**

---

## 🔑 Key Findings

### 📊 Top Price Predictors (Feature Importance)
```
1. 🥇 Car Age     → Older cars = significantly lower price
2. 🥈 Mileage     → Higher mileage = lower resale value
3. 🥉 Brand       → Luxury brands command premium pricing
4.    Engine Size → Larger engine = higher price tag
5.    Owner Count → More previous owners = lower value
```

### 💡 Business Insights
```
→ Every 1 year of age reduces car value by ~8-12%
→ Every 10,000 km driven adds to depreciation significantly
→ Electric and Hybrid cars command significant premium
→ Automatic transmission cars sell higher than manual
→ Single-owner cars are worth considerably more at resale
→ Brand premium: Luxury brands (BMW, Mercedes) 2x-4x standard brands
```

---

## 📁 Project Structure

```
CodeAlpha_DataScienceProject/
│
├── 📓 car_price_prediction_CodeAlpha.ipynb  ← Main ML notebook
├── 📊 car_price_dataset.csv                 ← Dataset (10,000 cars)
├── 🤖 car_price_model.pkl                   ← Saved trained model
└── 📄 README.md                             ← This file
```

---

## 🛠️ Technologies Used

```python
# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Preprocessing Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Evaluation
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model Saving
import joblib
```

---

## 🚀 How to Run

```bash
# Step 1: Clone repository
git clone https://github.com/Faizalik1/CodeAlpha_DataScienceProject.git
cd CodeAlpha_DataScienceProject

# Step 2: Install requirements
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter

# Step 3: Launch notebook
jupyter notebook car_price_prediction_CodeAlpha.ipynb

# Step 4: Run all cells
# Kernel → Restart & Run All ✅
```

---

## 🔮 Using the Saved Model

```python
import joblib
import pandas as pd

# Load the saved model (no retraining needed!)
model = joblib.load("car_price_model.pkl")

# Predict price for a new car
new_car = pd.DataFrame([{
    "Brand": "Toyota",
    "Car_Age": 2026 - 2019,    # 7 years old
    "Engine_Size": 2.5,
    "Fuel_Type": "Petrol",
    "Transmission": "Automatic",
    "Mileage": 45000,
    "Doors": 4,
    "Owner_Count": 1,
}])

predicted_price = model.predict(new_car)[0]
print(f"🚗 Predicted Car Price: ${predicted_price:,.2f}")
# Output: 🚗 Predicted Car Price: $18,450.00
```

---

## 🎓 Skills Demonstrated

- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering (Car_Age derived from Year)
- ✅ Sklearn Pipeline (Scaler + Encoder + Model in one)
- ✅ Multiple ML Algorithm Comparison (4 models)
- ✅ Model Evaluation (MAE, RMSE, R², 5-fold Cross-Validation)
- ✅ Feature Importance Analysis
- ✅ Model Serialization with joblib (.pkl file)
- ✅ Production-ready Prediction Function
- ✅ Business Insight Generation

---

## 🚀 Possible Next Steps

- [ ] Hyperparameter tuning with GridSearchCV or Optuna
- [ ] Add SHAP values for deeper model interpretability
- [ ] Include `Model` column with target encoding
- [ ] Deploy as Streamlit web app
- [ ] Build REST API with Flask or FastAPI

---

## 🗂️ More Projects

| Project | Description | Tools | Stars |
|---------|-------------|-------|-------|
| 🔍 [Customer Segmentation](https://github.com/Faizalik1) | K-Means on 53,503 insurance customers | Python, K-Means, Scikit-Learn | ⭐ |
| 🎬 [Netflix EDA](https://github.com/Faizalik1/CodeAlpha_DataAnalytics) | Content trend analysis 8,800+ titles | Python, Pandas, Seaborn | ⭐ |
| 🚗 Car Price Prediction | 4 ML models, R²=0.9995 | Python, XGBoost, Linear Regression | 📍 You are here |

---

## 🔗 Connect With Me

<p align="center">
  <a href="https://www.linkedin.com/in/faiz-ali-03365b217">
    <img src="https://img.shields.io/badge/LinkedIn-Faiz_Ali-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
  &nbsp;
  <a href="https://github.com/Faizalik1">
    <img src="https://img.shields.io/badge/GitHub-Faizalik1-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  &nbsp;
  <a href="mailto:faizalics1@gmail.com">
    <img src="https://img.shields.io/badge/Email-faizalics1@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>
</p>

---

<p align="center">
  <i>🚗 Project by Faiz Ali | CodeAlpha Data Science Internship | 2025</i>
  <br/>
  <i>⭐ If you found this project helpful, please give it a star! ⭐</i>
</p>
