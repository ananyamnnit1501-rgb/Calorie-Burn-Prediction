# 🔥 AI Fitness Calorie Burn Prediction System

An end-to-end Machine Learning project that predicts the number of calories burned during exercise using physiological and workout-related parameters. The project includes data preprocessing, feature engineering, model comparison, hyperparameter tuning, explainable AI using SHAP, and deployment through Streamlit.

---

##  Features

- Predicts calories burned based on user inputs
- Interactive Streamlit web application
- Explainable AI using SHAP
- Feature Engineering and Data Preprocessing
- Hyperparameter tuning with RandomizedSearchCV
- Model comparison between Random Forest and XGBoost
- Cross-validation for robust performance evaluation

---

## Dataset

The dataset contains **15,000+ fitness records** with workout and physiological attributes.

### Features

- Gender
- Age
- Height
- Weight
- Duration
- Heart Rate
- Body Temperature

### Engineered Features

- BMI
- Exercise Intensity
- Heat Stress Index

Target Variable

- Calories Burned

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Random Forest
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

##  Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Hyperparameter Tuning
7. Cross Validation
8. Model Evaluation
9. SHAP Explainability
10. Streamlit Deployment

---

##  Models Used

- Random Forest Regressor
- XGBoost Regressor

---

##  Model Performance

| Model | R² Score |
|--------|----------|
| Random Forest (Tuned) | **0.9959** |
| XGBoost | Compared for performance |

Evaluation Metrics

- R² Score
- MAE
- MSE
- RMSE

---

##  Explainable AI

The project uses **SHAP (SHapley Additive Explanations)** to interpret predictions.

Visualizations include:

- SHAP Summary Plot
- Feature Importance Plot
- Waterfall Plot
- Force Plot

---

## Streamlit Application

The web application allows users to:

- Enter workout details
- Predict calories burned instantly
- View an intuitive interface for real-time predictions

---

## Project Structure

AI-Fitness-Calorie-Prediction/

├── Dataset/

├── Models/

├── Notebook/

├── app.py

├── requirements.txt

├── calories_rf_model.pkl

├── calories_scaler.pkl

└── README.md






##  Future Improvements

- Workout Recommendation System
- Cloud Deployment
- REST API Integration
- Deep Learning Models
- User Authentication

