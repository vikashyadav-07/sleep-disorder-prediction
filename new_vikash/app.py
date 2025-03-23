import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load saved models and preprocessing objects
try:
    logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
    rf_model = pickle.load(open("random_forest_model.pkl", "rb"))
    gb_model = pickle.load(open("gradient_boosting_model.pkl", "rb"))
    dt_model = pickle.load(open("decision_tree_model.pkl", "rb"))
    svm_model = pickle.load(open("svm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    sleep_disorder_encoder = pickle.load(open("sleep_disorder_encoder.pkl", "rb"))
    occupation_encoder = pickle.load(open("occupation_encoder.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure all .pkl files are in the correct directory.")
    st.stop()

# Streamlit UI
st.title("🛌 Sleep Disorder Prediction")
st.markdown("""
This app predicts whether you might have a sleep disorder based on your lifestyle and health data.  
Select a model and enter your details below to get a prediction.  
**Note**: This is not a medical diagnosis. Consult a doctor for professional advice.
""")

# Model selection
st.subheader("Model Selection")
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Gradient Boosting", "Decision Tree", "SVM"])
model_dict = {
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Decision Tree": dt_model,
    "SVM": svm_model
}
selected_model = model_dict[model_choice]

# Display model performance (you can update these values based on your classification reports)
st.subheader("Model Performance (on Test Set)")
if model_choice == "Logistic Regression":
    st.write("- Accuracy: 92%")
    st.write("- F1-Score (weighted): [Update based on your classification report]")
elif model_choice == "Random Forest":
    st.write("- Accuracy: 88%")
    st.write("- F1-Score (weighted): [Update based on your classification report]")
elif model_choice == "Gradient Boosting":
    st.write("- Accuracy: 88%")
    st.write("- F1-Score (weighted): [Update based on your classification report]")
elif model_choice == "Decision Tree":
    st.write("- Accuracy: [Update based on your classification report]")
    st.write("- F1-Score (weighted): [Update based on your classification report]")
elif model_choice == "SVM":
    st.write("- Accuracy: [Update based on your classification report]")
    st.write("- F1-Score (weighted): [Update based on your classification report]")

# User inputs with exact unique values from the dataset
st.subheader("Enter Your Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.selectbox("Age", [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
    occupation = st.selectbox("Occupation", ["Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher"])
    sleep_duration = st.selectbox("Sleep Duration (hours)", [5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5])
    quality_of_sleep = st.selectbox("Quality of Sleep (1-10)", [4, 5, 6, 7, 8, 9])
    physical_activity_level = st.selectbox("Physical Activity Level (1-100)", [30, 32, 35, 40, 42, 45, 47, 50, 55, 60, 65, 70, 75, 80, 85, 90])

with col2:
    stress_level = st.selectbox("Stress Level (1-10)", [3, 4, 5, 6, 7, 8])
    bmi_category = st.selectbox("BMI Category", ["Normal Weight", "Overweight", "Obese"])
    heart_rate = st.selectbox("Heart Rate (beats per minute)", [65, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86])
    daily_steps = st.selectbox("Daily Steps", [3000, 3300, 3500, 3700, 4000, 4100, 4200, 4800, 5000, 5200, 5500, 5600, 6000, 6200, 6800, 7000, 7300, 7500, 8000, 10000])
    systolic_pressure = st.selectbox("Systolic Pressure (mmHg)", [115, 117, 118, 119, 120, 121, 122, 125, 126, 128, 129, 130, 131, 132, 135, 139, 140, 142])
    diastolic_pressure = st.selectbox("Diastolic Pressure (mmHg)", [75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 95])

# Encode categorical inputs to match training data
gender_encoded = 1 if gender == "Male" else 0
bmi_category_encoded = {"Normal Weight": 0, "Overweight": 1, "Obese": 2}[bmi_category]
occupation_encoded = occupation_encoder.transform([occupation])[0]

# Prepare input features
input_data = {
    'Gender': gender_encoded,
    'Age': age,
    'Occupation': occupation_encoded,
    'Sleep Duration': sleep_duration,
    'Quality of Sleep': quality_of_sleep,
    'Physical Activity Level': physical_activity_level,
    'Stress Level': stress_level,
    'BMI Category': bmi_category_encoded,
    'Heart Rate': heart_rate,
    'Daily Steps': daily_steps,
    'Systolic Pressure': systolic_pressure,
    'Diastolic Pressure': diastolic_pressure
}

# Create DataFrame with the exact same columns as feature_names
try:
    input_df = pd.DataFrame([input_data])[feature_names]
except KeyError as e:
    st.error(f"Feature mismatch: {e}. The input data does not contain all expected features.")
    st.stop()

# Scale input
try:
    input_scaled = scaler.transform(input_df)
except ValueError as e:
    st.error(f"Error scaling input: {e}. Ensure all features match the training data.")
    st.stop()

# Predict
st.subheader("Prediction")
if st.button("Predict Sleep Disorder"):
    try:
        prediction = selected_model.predict(input_scaled)
        predicted_label = sleep_disorder_encoder.inverse_transform(prediction)[0]

        if predicted_label == "None":
            st.success("✅ No sleep disorder detected.")
        else:
            st.error(f"⚠️ Possible **{predicted_label}** sleep disorder. Consult a doctor.")

        # Display prediction probability (if supported by the model)
        if hasattr(selected_model, "predict_proba"):
            prob = selected_model.predict_proba(input_scaled)[0]
            st.write("**Prediction Probabilities:**")
            prob_df = pd.DataFrame({
                "Sleep Disorder": sleep_disorder_encoder.classes_,
                "Probability (%)": [f"{p*100:.2f}" for p in prob]
            })
            st.table(prob_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Display feature importance for Random Forest
if model_choice == "Random Forest":
    st.subheader("Feature Importance (Random Forest)")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.table(importance_df)

# User feedback
st.subheader("Feedback")
feedback = st.text_area("Did the prediction match your expectations? Please provide feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")