# 🚗 Car Price Prediction — Machine Learning Project

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=2196F3&center=true&vCenter=true&width=500&lines=Car+Price+Prediction+%F0%9F%9A%97;Linear+Regression+%7C+XGBoost+%7C+Random+Forest;R%C2%B2+%3D+0.9995+Accuracy+%F0%9F%94%A5;CodeAlpha+Data+Science+Internship" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Internship-CodeAlpha-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Track-Data%20Science-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Model%20Accuracy-R²%3D0.9995-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-10%2C000%20Cars-orange?style=flat-square"/>
</p>

---

## 📋 Project Overview

A complete **end-to-end machine learning project** that predicts used car prices based on vehicle attributes. This project compares **4 regression algorithms**, builds a full preprocessing pipeline, saves the trained model, and provides a ready-to-use prediction function.

**Internship:** CodeAlpha — Data Science Track
**Author:** Faiz Ali

---

## 🎯 Business Problem

The used car market faces a major pricing challenge:
- 🤔 Buyers don't know if they're overpaying
- 🤔 Sellers don't know the fair market value
- 🤔 Dealerships need automated pricing tools

**Solution:** Build an ML model that predicts car prices accurately based on key vehicle features — enabling data-driven pricing decisions.

---

## 📊 Dataset Overview

**Size:** 10,000 cars | 10 columns | No missing values | No duplicates

| Column | Description | Example |
|--------|-------------|---------|
| Brand | Car manufacturer | Toyota, BMW, Honda |
| Model | Specific car model | Corolla, 3 Series |
| Year | Manufacturing year | 2019 |
| Engine_Size | Engine size in liters | 2.5 |
| Fuel_Type | Fuel category | Diesel, Petrol, Hybrid, Electric |
| Transmission | Gearbox type | Manual, Automatic, Semi-Auto |
| Mileage | Total km driven | 45,000 |
| Doors | Number of doors | 4 |
| Owner_Count | Number of previous owners | 1 |
| **Price** | **🎯 Target variable** | $25,000 |

---

## 🔬 Complete Analysis Pipeline

```
1️⃣  DATA LOADING        → Load CSV, inspect shape, types, stats
2️⃣  EDA                 → Price distribution, correlations, category analysis
3️⃣  FEATURE ENGINEERING → Car_Age derived from Year, dropped Model column
4️⃣  PREPROCESSING       → StandardScaler + OneHotEncoder Pipeline
5️⃣  MODEL TRAINING      → 4 algorithms trained and compared
6️⃣  EVALUATION          → MAE, RMSE, R², Cross-validation
7️⃣  FEATURE IMPORTANCE  → Top predictors identified
8️⃣  PREDICTION FUNCTION → predict_car_price() for new cars
9️⃣  MODEL SAVING        → Exported as car_price_model.pkl
```

---

## 🤖 Models Compared

| 🏆 Model | MAE | RMSE | R² Score |
|---------|-----|------|----------|
| 🥇 **Linear Regression** | **19.71** | **64.77** | **0.9995** |
| 🥈 XGBoost | 116.27 | 150.95 | 0.9975 |
| 🥉 Gradient Boosting | 169.55 | 216.66 | 0.9949 |
| 4️⃣ Random Forest | 246.75 | 318.03 | 0.9890 |

> 💡 **Why did Linear Regression win?**
> The relationship between car price and key features (car age + mileage) is largely **linear** in this dataset. Simpler models generalize better when the underlying pattern is linear — proving that the best model is not always the most complex one!

---

## 📈 Key Findings

### 🔑 Top Price Predictors
```
1. Car Age    → Older cars = significantly lower price
2. Mileage    → Higher mileage = lower price
3. Brand      → Luxury brands command premium pricing
4. Engine Size → Larger engine = higher price
5. Owner Count → More owners = lower resale value
```

### 💡 Business Insights
```
→ A 1-year increase in car age reduces value by ~8-12%
→ Every 10,000 km adds to depreciation significantly
→ Electric/Hybrid cars command premium pricing
→ Automatic transmission cars sell higher than manual
→ Single-owner cars are worth significantly more
```

---

## 📁 Project Structure

```
CodeAlpha_DataScienceProject/
│
├── 📓 car_price_prediction_CodeAlpha.ipynb   ← Main notebook
├── 📊 car_price_dataset.csv                  ← Dataset (10,000 cars)
├── 🤖 car_price_model.pkl                    ← Saved trained model
└── 📄 README.md                              ← This file
```

---

## 🛠️ Technologies Used

```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Visualizations
import seaborn as sns        # Statistical plots
from sklearn.linear_model import LinearRegression  # Primary model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib                # Model saving/loading
```

---

## 🚀 How to Run

### Step 1: Clone Repository
```bash
git clone https://github.com/Faizalik1/CodeAlpha_DataScienceProject.git
cd CodeAlpha_DataScienceProject
```

### Step 2: Install Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

### Step 3: Run Notebook
```bash
jupyter notebook car_price_prediction_CodeAlpha.ipynb
```

### Step 4: Run All Cells
```
Kernel → Restart & Run All ✅
```

---

## 🔮 Using the Saved Model

```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("car_price_model.pkl")

# Predict price for a new car
new_car = pd.DataFrame([{
    "Brand": "Toyota",
    "Car_Age": 2026 - 2019,   # 7 years old
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
- ✅ Feature Engineering (Car_Age derivation)
- ✅ Sklearn Pipeline (Scaler + Encoder + Model)
- ✅ Multiple ML Algorithm Comparison
- ✅ Model Evaluation (MAE, RMSE, R², Cross-Validation)
- ✅ Feature Importance Analysis
- ✅ Model Serialization (joblib .pkl)
- ✅ Production-ready Prediction Function
- ✅ Business Insight Generation

---

## 🚀 Possible Next Steps

- [ ] Hyperparameter tuning with GridSearchCV or Optuna
- [ ] Add SHAP values for deeper model interpretability
- [ ] Include `Model` column with target/frequency encoding
- [ ] Deploy as Streamlit web app
- [ ] Build REST API with Flask/FastAPI

---

## 🔗 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Faiz_Ali-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/faiz-ali-03365b217)
[![GitHub](https://img.shields.io/badge/GitHub-Faizalik1-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Faizalik1)
[![Email](https://img.shields.io/badge/Email-faizalics1@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:faizalics1@gmail.com)

---

<p align="center">
  <i>🚗 Project by Faiz Ali | CodeAlpha Data Science Internship | 2025</i>
</p>

<p align="center">
  <i>⭐ If you found this project helpful, please give it a star! ⭐</i>
</p>



---
Project by Faiz Ali | CodeAlpha Data Science Internship
