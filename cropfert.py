import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load datasets
crop_df = pd.read_csv("Crop_recommendation.csv")
fertilizer_df = pd.read_csv("Fertilizer Prediction.csv")

# Encode categorical columns in fertilizer dataset
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_df['Soil Type'] = soil_encoder.fit_transform(fertilizer_df['Soil Type'])
fertilizer_df['Crop Type'] = crop_encoder.fit_transform(fertilizer_df['Crop Type'])

# Encode crop labels
crop_labels = crop_df['label'].unique()
crop_encoder.fit(crop_labels)

# Prepare Crop Recommendation Model
X_crop = crop_df.drop(columns=['label'])
y_crop = crop_df['label']
crop_model = RandomForestClassifier()
crop_model.fit(X_crop, y_crop)

# Prepare Fertilizer Recommendation Model
X_fertilizer = fertilizer_df.drop(columns=['Fertilizer Name'])
y_fertilizer = fertilizer_df['Fertilizer Name']
fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(X_fertilizer, y_fertilizer)

# Streamlit App
st.set_page_config(page_title="Crop and Fertilizer Recommendation", page_icon="🌱", layout="wide")

# Function to set background image
def set_background(image_url):
    st.markdown(f"""
        <style>
            .stApp {{
                background: url({image_url}) no-repeat center center fixed;
                background-size: cover;
            }}
        </style>
    """, unsafe_allow_html=True)

# Sidebar Dashboard Selection
st.sidebar.title("🌱 Dashboard")
option = st.sidebar.radio("Choose an option:", ["Dashboard", "Recommend Crop Only", "Recommend Fertilizer Only", "Recommend Both"])

# Change background based on selection
if option == "Dashboard":
    set_background("https://example.com/dashboard_image.jpg")
elif option in ["Recommend Crop Only", "Recommend Fertilizer Only", "Recommend Both"]:
    set_background("https://images.app.goo.gl/HEysLQUvEThGncwYA")

st.title("🌾 Crop and Fertilizer Recommendation System")

if option != "Dashboard":
    st.header("🌿 Enter Input Values")
    N = st.number_input("Nitrogen", min_value=0, max_value=200, value=50, key="crop_nitrogen")
    P = st.number_input("Phosphorus", min_value=0, max_value=200, value=50, key="crop_phosphorus")
    K = st.number_input("Potassium", min_value=0, max_value=200, value=50, key="crop_potassium")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=50.0, value=25.0, key="crop_temperature")
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=50.0, key="crop_humidity")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, key="crop_ph")
    rainfall = st.number_input("Rainfall", min_value=0.0, max_value=500.0, value=100.0, key="crop_rainfall")
    
    soil_type = st.selectbox("Soil Type", soil_encoder.classes_, key="fertilizer_soil_type")
    soil_encoded = soil_encoder.transform([soil_type])[0]
    
    crop_type = None
    if option in ["Recommend Fertilizer Only", "Recommend Both"]:
        crop_type = st.selectbox("Crop Type", crop_labels, key="fertilizer_crop_type")
        crop_encoded = crop_encoder.transform([crop_type])[0]
    
    if st.button("🌱 Get Recommendation"):
        set_background("https://example.com/after_recommendation.jpg")
        
        # Crop Prediction
        if option in ["Recommend Crop Only", "Recommend Both"]:
            crop_pred = crop_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])[0]
            st.success(f"🌾 Recommended Crop: {crop_pred}")
            if option == "Recommend Both":
                crop_encoded = crop_encoder.transform([crop_pred])[0]  # Update crop_encoded for fertilizer prediction
        
        # Fertilizer Prediction
        if option in ["Recommend Fertilizer Only", "Recommend Both"] and crop_encoded is not None:
            try:
                # Make sure the correct input features are passed in the right order
                fertilizer_pred = fertilizer_model.predict([[temperature, humidity, 50, soil_encoded, crop_encoded, N, K, P]])[0]
                st.success(f"🧪 Recommended Fertilizer: {fertilizer_pred}")
            except Exception as e:
                st.error(f"An error occurred while predicting the fertilizer: {str(e)}")
