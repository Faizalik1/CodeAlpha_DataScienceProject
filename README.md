# 🚗 Car Price Prediction



![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)




![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)




![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)




![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)




![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge&logo=xgboost&logoColor=white)




![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)




![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)



An end-to-end machine learning project that predicts used car prices based on vehicle attributes like brand, year, mileage, engine size, fuel type, transmission, and ownership history. Built as part of the CodeAlpha Data Science Internship (Task 3).

## 📂 Project Structure

car_price_project/
├── car_price_prediction_CodeAlpha.ipynb   # Main notebook (EDA + modeling + evaluation)
├── car_price_dataset.csv                  # Dataset (10,000 cars, 10 columns)
├── car_price_model.pkl                    # Saved trained model pipeline
└── README.md                              # This file

## 📊 Dataset Overview

| Column | Description |
|---|---|
| Brand | Car manufacturer (e.g. Toyota, BMW, Honda) |
| Model | Specific car model |
| Year | Manufacturing year |
| Engine_Size | Engine size in liters |
| Fuel_Type | Diesel / Hybrid / Electric / Petrol |
| Transmission | Manual / Automatic / Semi-Automatic |
| Mileage | Total mileage driven |
| Doors | Number of doors |
| Owner_Count | Number of previous owners |
| **Price** | **Target variable** — car price (USD) |

10,000 rows, no missing values, no duplicates.

## 🔧 What the Notebook Does

1. **Data Loading & Inspection** — shape, types, summary stats
2. **Exploratory Data Analysis (EDA)**
   - Price distribution (histogram + boxplot)
   - Categorical feature counts (Brand, Fuel Type, Transmission)
   - Average price by category
   - Numerical features vs. price (scatter plots)
   - Correlation heatmap
3. **Feature Engineering**
   - Derived Car_Age from Year
   - Dropped high-cardinality Model column
4. **Preprocessing Pipeline**
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features
   - Wrapped in a ColumnTransformer + sklearn Pipeline
5. **Model Training & Comparison**
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
6. **Evaluation**
   - MAE, RMSE, R² for each model
   - 5-fold cross-validation on the best model
   - Predicted vs. actual price plot
   - Residual analysis
7. **Feature Importance** — top predictors visualized
8. **Prediction Function** — predict_car_price() for new/unseen cars
9. **Model Saving** — best pipeline exported as car_price_model.pkl

## 🏆 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **19.71** | **64.77** | **0.9995** |
| XGBoost | 116.27 | 150.95 | 0.9975 |
| Gradient Boosting | 169.55 | 216.66 | 0.9949 |
| Random Forest | 246.75 | 318.03 | 0.9890 |

Linear Regression performed best — the relationship between price and the key features (especially car age and mileage) is largely linear in this dataset, so the simpler model generalizes extremely well.

**Top predictors:** Car age and mileage explain most of the price variation, with brand contributing secondary signal.

> **Note:** The very high R² (0.9995) reflects the structured, low-noise nature of this dataset rather than a real-world used-car market, where pricing is typically noisier due to condition, location, and demand factors not captured here.

## ▶️ How to Run

1. Make sure car_price_dataset.csv is in the same folder as the notebook.
2. Install dependencies: pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
3. Open and run the notebook top to bottom: jupyter notebook car_price_prediction_CodeAlpha.ipynb

## 🔮 Using the Trained Model

import joblib
import pandas as pd

model = joblib.load("car_price_model.pkl")

new_car = pd.DataFrame([{
    "Brand": "Toyota",
    "Car_Age": 2026 - 2019,
    "Engine_Size": 2.5,
    "Fuel_Type": "Petrol",
    "Transmission": "Automatic",
    "Mileage": 45000,
    "Doors": 4,
    "Owner_Count": 1,
}])

predicted_price = model.predict(new_car)[0]
print(f"Predicted Price: ${predicted_price:,.2f}")

## 🚀 Possible Next Steps

- Hyperparameter tuning (GridSearchCV / Optuna) for tree-based models
- Target/frequency encoding for the Model column instead of dropping it
- Deploy as a simple Streamlit or Flask web app
- Add SHAP values for deeper model interpretability

## 🛠️ Tech Stack



![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)




![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)




![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)




![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)




![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=flat-square&logo=xgboost&logoColor=white)




![Seaborn](https://img.shields.io/badge/Seaborn-4051B5?style=flat-square&logo=python&logoColor=white)




![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)



---
Project by Faiz Ali | CodeAlpha Data Science Internship