import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model, scaler, and price transformer
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
price_transformer = joblib.load('price_transformer.pkl')

# Streamlit app
st.title("California House Price Prediction")
st.write("Enter details to predict house price based on California 1990 Census data.")

# Input fields for all features used in the model
bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=2000, step=1, value=500)
square_footage = st.number_input("Total Rooms (proxy for Square Footage)", min_value=100, max_value=10000, step=100, value=2000)
median_income = st.number_input("Median Income (in tens of thousands)", min_value=0.0, max_value=15.0, step=0.1, value=3.0)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, step=0.1, value=37.7)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, step=0.1, value=-122.4)
population = st.number_input("Population in Block", min_value=0, max_value=10000, step=100, value=1000)
households = st.number_input("Households in Block", min_value=0, max_value=5000, step=100, value=400)
ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'])

# Prediction button
if st.button("Predict Price"):
    # Prepare input data with numerical features
    input_data = pd.DataFrame([[bedrooms, square_footage, median_income, latitude, longitude, population, households]], 
                              columns=['Bedrooms', 'SquareFootage', 'MedianIncome', 'Latitude', 'Longitude', 'Population', 'Households'])
    
    # Compute DistToSF and interaction term
    input_data['DistToSF'] = np.sqrt((input_data['Latitude'] - 37.7749)**2 + (input_data['Longitude'] - (-122.4194))**2)
    input_data['Income_DistToSF'] = input_data['MedianIncome'] * input_data['DistToSF']
    
    # Scale numerical features
    numerical_cols = ['Bedrooms', 'SquareFootage', 'MedianIncome', 'Latitude', 'Longitude', 'Population', 'Households', 'DistToSF', 'Income_DistToSF']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # Add all encoded OceanProximity columns (matching training data)
    ocean_cols = ['OceanProximity_INLAND', 'OceanProximity_ISLAND', 'OceanProximity_NEAR BAY', 'OceanProximity_NEAR OCEAN']
    for col in ocean_cols:
        input_data[col] = 0
    
    # Set the appropriate column based on user input
    if ocean_proximity == 'INLAND':
        input_data['OceanProximity_INLAND'] = 1
    elif ocean_proximity == 'ISLAND':
        input_data['OceanProximity_ISLAND'] = 1
    elif ocean_proximity == 'NEAR BAY':
        input_data['OceanProximity_NEAR BAY'] = 1
    elif ocean_proximity == 'NEAR OCEAN':
        input_data['OceanProximity_NEAR OCEAN'] = 1
    # If ocean_proximity is '<1H OCEAN', leave all columns as 0 (since it was dropped during training)
    
    # Ensure the order of columns matches the training data
    expected_columns = ['Bedrooms', 'SquareFootage', 'MedianIncome', 'Latitude', 'Longitude', 'Population', 'Households', 'DistToSF', 'Income_DistToSF', 'OceanProximity_INLAND', 'OceanProximity_ISLAND', 'OceanProximity_NEAR BAY', 'OceanProximity_NEAR OCEAN']
    input_data = input_data[expected_columns]
    
    # Make prediction
    predicted_price_transformed = model.predict(input_data)[0]
    predicted_price = price_transformer.inverse_transform([[predicted_price_transformed]])[0][0]
    
    # Display result
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")

st.write("Model uses XGBoost with features: Bedrooms, Square Footage, Median Income, Location, Population, Households, and Ocean Proximity.")