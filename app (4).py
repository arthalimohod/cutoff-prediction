import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and features
@st.cache_resource
def load_resources():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

try:
    model, model_features = load_resources()
    
    st.title("🎯 University Cutoff Predictor")
    st.write("Predict the cutoff values based on the trained Random Forest Regressor.")
    
    # User Inputs
    year = st.selectbox("Year", [2024, 2025, 2026])
    
    # Extracted unique categorical choices from original feature columns
    # In a production scenario, you would dynamically pass these lists from your original dataset
    university_id = st.text_input("University ID (e.g., banaras-hindu-university)", value="banaras-hindu-university")
    seat_type = st.text_input("Seat Type (e.g., general, Delhi region)", value="general")
    figure_kind = st.selectbox("Figure Kind", ["score", "rank"])
    basis = st.text_input("Basis (e.g., official)", value="official")
    
    if st.button("Predict Cutoff"):
        # Construct input DataFrame matching original features before one-hot encoding
        input_data = pd.DataFrame([{
            'year': year,
            'university_id': university_id,
            'seat_type': seat_type,
            'figure_kind': figure_kind,
            'basis': basis
        }])
        
        # Apply one-hot encoding
        input_encoded = pd.get_dummies(input_data)
        
        # Align with original features (fill missing columns with 0)
        aligned_input = pd.DataFrame(0, index=[0], columns=model_features)
        aligned_input['year'] = year
        
        for col in input_encoded.columns:
            if col in aligned_input.columns:
                aligned_input[col] = input_encoded[col].values[0]
                
        # Make prediction
        prediction = model.predict(aligned_input)[0]
        st.success(f"✨ Predicted Cutoff: {prediction:.2f}")
        
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'rf_model.pkl' and 'model_features.pkl' are in the same directory as the app.")
