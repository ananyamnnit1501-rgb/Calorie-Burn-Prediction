# ============================================================
# AI CALORIE & WORKOUT RECOMMENDATION SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Fitness Calorie Predictor",
    page_icon="🔥",
    layout="wide"
)

# ============================================================
# LOAD MODELS
# ============================================================

rf_model = joblib.load("D:\streamlit calorie project\calories_rf_model.pkl")

scaler = joblib.load("D:\streamlit calorie project\calories_scaler.pkl")

reverse_model = joblib.load(
    "reverse_duration_model.pkl"
)

# ============================================================
# TITLE
# ============================================================

st.markdown("""
#  AI Fitness Calorie Prediction Platform

Industry-Level Explainable AI Fitness System
""")

# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs([
    " Calories Prediction",
    " Workout Goal Planner"
])

# ============================================================
# TAB 1 — CALORIE PREDICTION
# ============================================================

with tab1:

    st.header("Calories Burned Prediction")

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.slider(
            "Age",
            10,
            80,
            25
        )

        height = st.slider(
            "Height (cm)",
            120,
            220,
            170
        )

        weight = st.slider(
            "Weight (kg)",
            30,
            150,
            70
        )

    with col2:

        duration = st.slider(
            "Workout Duration (minutes)",
            5,
            180,
            45
        )

        heart_rate = st.slider(
            "Heart Rate",
            60,
            200,
            110
        )

        body_temp = st.slider(
            "Body Temperature",
            35.0,
            42.0,
            40.0
        )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    bmi = weight / ((height / 100) ** 2)

    exercise_intensity = (
        duration * heart_rate
    )

    heat_stress = (
        duration * body_temp
    )

    gender_encoded = 1 if gender == "Male" else 0

    # --------------------------------------------------------
    # SHOW ENGINEERED FEATURES
    # --------------------------------------------------------

    st.subheader("Derived Features")

    c1, c2, c3 = st.columns(3)

    c1.metric("BMI", round(bmi, 2))

    c2.metric(
        "Exercise Intensity",
        round(exercise_intensity, 2)
    )

    c3.metric(
        "Heat Stress",
        round(heat_stress, 2)
    )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button("Predict Calories Burned"):

        input_df = pd.DataFrame({

            'Gender': [gender_encoded],

            'Age': [age],

            'Height': [height],

            'Weight': [weight],

            'Duration': [duration],

            'Heart_Rate': [heart_rate],

            'Body_Temp': [body_temp],

            'BMI': [bmi],

            'Exercise_Intensity': [
                exercise_intensity
            ],

            'Heat_Stress': [heat_stress]
        })

        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        input_scaled = scaler.transform(input_df)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = rf_model.predict(
            input_scaled
        )[0]

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        st.success(
            f" Estimated Calories Burned: "
            f"{prediction:.2f} kcal"
        )

        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.subheader("Explainable AI (SHAP)")

        explainer = shap.TreeExplainer(rf_model)

        shap_values = explainer.shap_values(
            input_scaled
        )

        # ----------------------------------------------------
        # SHAP BAR VALUES
        # ----------------------------------------------------

        shap_df = pd.DataFrame({

            'Feature': input_df.columns,

            'SHAP Value': shap_values[0]
        })

        shap_df = shap_df.sort_values(
            by='SHAP Value',
            ascending=False
        )

        st.dataframe(shap_df)

        # ----------------------------------------------------
        # SHAP PLOT
        # ----------------------------------------------------

        fig, ax = plt.subplots(figsize=(8,5))

        shap_df.plot(
            kind='barh',
            x='Feature',
            y='SHAP Value',
            ax=ax
        )

        plt.title("Feature Contribution")

        st.pyplot(fig)

# ============================================================
# TAB 2 — WORKOUT GOAL PLANNER
# ============================================================

with tab2:

    st.header("Workout Goal Planner")

    st.write(
        "Enter your target calories to burn."
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        goal_gender = st.selectbox(
            "Gender ",
            ["Male", "Female"]
        )

        goal_age = st.slider(
            "Age ",
            10,
            80,
            25
        )

        goal_height = st.slider(
            "Height ",
            120,
            220,
            170
        )

    with col2:

        goal_weight = st.slider(
            "Weight ",
            30,
            150,
            70
        )

        target_calories = st.slider(
            "Target Calories",
            50,
            2000,
            400
        )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    goal_bmi = (
        goal_weight /
        ((goal_height / 100) ** 2)
    )

    goal_gender_encoded = (
        1 if goal_gender == "Male" else 0
    )

    # --------------------------------------------------------
    # PREDICT WORKOUT PLAN
    # --------------------------------------------------------

    if st.button("Generate Workout Plan"):

        reverse_input = pd.DataFrame({

            'Calories': [target_calories],

            'Gender': [goal_gender_encoded],

            'Age': [goal_age],

            'Height': [goal_height],

            'Weight': [goal_weight],

            'BMI': [goal_bmi]
        })

        predicted_duration = reverse_model.predict(
            reverse_input
        )[0]

        # Estimated HR recommendation
        suggested_hr = 110 + (
            target_calories / 50
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        st.success(
            f" Required Workout Duration: "
            f"{predicted_duration:.1f} minutes"
        )

        st.info(
            f" Suggested Heart Rate: "
            f"{suggested_hr:.0f} bpm"
        )

        st.metric(
            "BMI",
            round(goal_bmi, 2)
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
---
Built with  using:
- Random Forest Regression
- Explainable AI (SHAP)
- Streamlit
- Scikit-Learn
""")
