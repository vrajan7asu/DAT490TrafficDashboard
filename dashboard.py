import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_folium import folium_static, st_folium
import folium
import json

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('rf_final_model.joblib')

rf_final = load_model()

# Function to predict severity based on user input
def predict_severity(model, input_data):
    input_df = pd.DataFrame([input_data])
    return model.predict(input_df)

# Title of the dashboard
st.title("Traffic Accident Severity Prediction Dashboard")

# Create a map centered on the US
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Use st_folium to render the map and capture clicks
st.subheader("Click on the Map Twice to Set a Start Location and End Location. Scroll Down and Hit Predict to Get the Model Predictions")
map_data = st_folium(m, width=700, height=500)

# Initialize session state variables
if 'start_lat' not in st.session_state:
    st.session_state.start_lat = 0.0
    st.session_state.start_lng = 0.0
    st.session_state.end_lat = 0.0
    st.session_state.end_lng = 0.0
    st.session_state.clicks = 0

# Process map clicks
if map_data['last_clicked']:
    st.session_state.clicks += 1
    if st.session_state.clicks % 2 == 1:  # Odd click - start point
        st.session_state.start_lat = map_data['last_clicked']['lat']
        st.session_state.start_lng = map_data['last_clicked']['lng']
    else:  # Even click - end point
        st.session_state.end_lat = map_data['last_clicked']['lat']
        st.session_state.end_lng = map_data['last_clicked']['lng']

# Display coordinate inputs
st.subheader("Coordinates")
col1, col2 = st.columns(2)
with col1:
    start_lat = st.number_input("Start Latitude", value=st.session_state.start_lat, format="%.6f", key="start_lat_input")
    start_lng = st.number_input("Start Longitude", value=st.session_state.start_lng, format="%.6f", key="start_lng_input")
with col2:
    end_lat = st.number_input("End Latitude", value=st.session_state.end_lat, format="%.6f", key="end_lat_input")
    end_lng = st.number_input("End Longitude", value=st.session_state.end_lng, format="%.6f", key="end_lng_input")

# Sidebar for other input features
st.sidebar.header("Input Features")

distance = st.sidebar.number_input("Distance (mi)", min_value=0.0, value=1.0)
temperature = st.sidebar.slider("Temperature (F)", min_value=-30, max_value=120, value=60)
wind_chill = st.sidebar.slider("Wind Chill (F)", min_value=-50, max_value=120, value=55)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=50)
pressure = st.sidebar.slider("Pressure (in)", min_value=25.0, max_value=35.0, value=30.0)
visibility = st.sidebar.slider("Visibility (mi)", min_value=0.0, max_value=10.0, value=5.0)
wind_speed = st.sidebar.slider("Wind Speed (mph)", min_value=0, max_value=100, value=10)
precipitation = st.sidebar.slider("Precipitation (in)", min_value=0.0, max_value=10.0, value=0.0)
population = st.sidebar.number_input("Population", min_value=0, value=100000)

# Organize user input into a dictionary
input_data = {
    'Start_Lat': start_lat,
    'Start_Lng': start_lng,
    'End_Lat': end_lat,
    'End_Lng': end_lng,
    'Distance(mi)': distance,
    'Temperature(F)': temperature,
    'Wind_Chill(F)': wind_chill,
    'Humidity(%)': humidity,
    'Pressure(in)': pressure,
    'Visibility(mi)': visibility,
    'Wind_Speed(mph)': wind_speed,
    'Precipitation(in)': precipitation,
    'Population': population
}

# Button to predict severity based on user inputs
if st.button("Predict Severity"):
    if start_lat == 0 or end_lat == 0:
        st.error("Please select both start and end points on the map.")
    else:
        severity_prediction = predict_severity(rf_final, input_data)
        st.subheader(f"Predicted Severity: {severity_prediction[0]}")

        # Show probabilities for all classes
        probabilities = rf_final.predict_proba(pd.DataFrame([input_data]))
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame(probabilities, columns=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'])
        st.write(prob_df)
