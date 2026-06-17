import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load the best model and the column names from X_train (for consistent feature order)
try:
    model_path = 'tourism_project/model_building/best_tourism_classifier.pkl'
    model = joblib.load(model_path)

    X_train_path = 'tourism_project/model_building/X_train.csv'
    X_train_columns = pd.read_csv(X_train_path).columns
except FileNotFoundError:
    st.error("Error: Model or X_train_columns file not found. Please ensure `best_tourism_classifier.pkl` and `X_train.csv` exist in `tourism_project/model_building/`.")
    st.stop()

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

st.set_page_config(page_title="Tourism Package Prediction", page_icon=":airplane:")

st.title(":airplane: Wellness Tourism Package Prediction")
st.markdown("Predict whether a customer will purchase the Wellness Tourism Package.")

# --- User Input Features ---
st.header("Customer Information")

# Input fields matching the model's features (excluding 'CustomerID' and 'ProdTaken')
# Numerical inputs
age = st.slider('Age', 18, 90, 35)
monthly_income = st.number_input('Monthly Income', 5000, 200000, 30000)
number_of_person_visiting = st.slider('Number of Persons Visiting', 1, 10, 2)
number_of_trips = st.slider('Number of Trips Annually', 0, 20, 2)
number_of_children_visiting = st.slider('Number of Children Visiting (under 5)', 0, 5, 0)
duration_of_pitch = st.slider('Duration of Pitch (minutes)', 1, 60, 10)
pitch_satisfaction_score = st.slider('Pitch Satisfaction Score (1-5)', 1, 5, 3)

# Categorical inputs - These need to be one-hot encoded
type_of_contact = st.selectbox('Type of Contact', ['Company Invited', 'Self Inquiry'])
city_tier = st.selectbox('City Tier', [1, 2, 3])
occupation = st.selectbox('Occupation', ['Salaried', 'Freelancer', 'Small Business', 'Large Business'])
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
preferred_property_star = st.selectbox('Preferred Property Star', [3, 4, 5])
product_pitched = st.selectbox('Product Pitched', ['Basic', 'Standard', 'Deluxe', 'Super Deluxe'])
designation = st.selectbox('Designation', ['Manager', 'Executive', 'Senior Manager', 'VP'])
number_of_followups = st.slider('Number of Follow-ups', 0, 10, 2)

# Binary inputs (already 0/1)
passport = st.checkbox('Has Passport')
own_car = st.checkbox('Owns Car')

# --- Preprocessing User Input ---
def preprocess_input(input_data):
    # Create a DataFrame from the input
    data = pd.DataFrame([input_data])

    # Apply the same preprocessing steps as in training

    # Gender: Male=1, Female=0
    data['Gender'] = data['Gender'].replace({'Male': 1, 'Female': 0})

    # Passport and OwnCar are already binary from Streamlit checkboxes
    data['Passport'] = data['Passport'].astype(int)
    data['OwnCar'] = data['OwnCar'].astype(int)

    # One-hot encode categorical features
    categorical_cols_to_encode = [
        'TypeofContact', 'Occupation', 'MaritalStatus', 'ProductPitched', 'Designation'
    ]
    
    # Apply one-hot encoding using pd.get_dummies
    data = pd.get_dummies(data, columns=categorical_cols_to_encode, drop_first=True)

    # Align columns with X_train - add missing columns, remove extra ones
    # Reindex the input data to match the order of X_train_columns
    missing_cols = set(X_train_columns) - set(data.columns)
    for c in missing_cols:
        data[c] = 0  # Add missing columns with default value 0
    data = data[X_train_columns] # Ensure the order of columns is the same as in training data

    return data


if st.button('Predict Purchase'):
    input_data = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'NumberOfTrips': number_of_trips,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'DurationOfPitch': duration_of_pitch,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'NumberOfFollowups': number_of_followups,
        'TypeofContact': type_of_contact,
        'CityTier': city_tier,
        'Occupation': occupation,
        'Gender': gender,
        'MaritalStatus': marital_status,
        'PreferredPropertyStar': preferred_property_star,
        'ProductPitched': product_pitched,
        'Designation': designation,
        'Passport': int(passport),
        'OwnCar': int(own_car)
    }

    # Create a dummy 'Unnamed: 0' column which was present in the original X_train
    # This is a temporary fix if 'Unnamed: 0' was dropped and not truly part of features.
    # In a robust pipeline, this would be handled consistently in preprocessing.
    input_df = pd.DataFrame([input_data])
    if 'Unnamed: 0' in X_train_columns and 'Unnamed: 0' not in input_df.columns:
        input_df['Unnamed: 0'] = 0 # or any placeholder value

    processed_input = preprocess_input(input_df)

    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"The customer is predicted to **PURCHASE** the package! (Probability: {prediction_proba[0]:.2f})")
    else:
        st.info(f"The customer is predicted **NOT to purchase** the package. (Probability: {prediction_proba[0]:.2f})")

    st.markdown("---")
    st.markdown("**Note:** This is a predictive model's output and should be used as a guide. Further analysis and business context are recommended.")

