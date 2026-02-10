from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI
app = FastAPI()

# Load the trained model
model = joblib.load('heart_model.pkl')

# Define the input data structure
class PatientData(BaseModel):
    gender: int
    age: int  # age in days
    height: int # in cm
    weight: float # in kg
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    active: int

# Helper functions for Preprocessing
def get_age_bin(age_days):
    age_years = age_days / 365.25
    bins = [0, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    for i in range(len(bins)-1):
        if bins[i] <= age_years < bins[i+1]:
            return i
    return 6

def get_bmi_class(weight, height_cm):
    bmi = weight / ((height_cm / 100) ** 2)
    if bmi < 18.5: return 1
    elif bmi < 24.9: return 2
    elif bmi < 29.9: return 3
    elif bmi < 34.9: return 4
    elif bmi < 39.9: return 5
    else: return 6

def get_map_class(ap_hi, ap_lo):
    map_val = (ap_hi + 2 * ap_lo) / 3
    if map_val < 69.9: return 1
    elif map_val < 79.9: return 2
    elif map_val < 89.9: return 3
    elif map_val < 99.9: return 4
    elif map_val < 109.9: return 5
    elif map_val < 119.9: return 6
    else: return 7

@app.post("/predict")
def predict(data: PatientData):
    # 1. Preprocess raw data into features
    age_bin = get_age_bin(data.age)
    bmi_class = get_bmi_class(data.weight, data.height)
    map_class = get_map_class(data.ap_hi, data.ap_lo)
    
    # 2. Arrange features in the correct order for the model
    # Order: [gender, age_bin, BMI_Class, MAP_Class, cholesterol, gluc, smoke, active]
    features = np.array([[
        data.gender, age_bin, bmi_class, map_class, 
        data.cholesterol, data.gluc, data.smoke, data.active
    ]])

    # 3. Get Prediction and Probability
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    # 4. Determine status message
    if probability > 70:
        message = "High Risk: Please consult a specialist immediately."
    elif probability > 40:
        message = "Moderate Risk: Consider lifestyle changes and check-ups."
    else:
        message = "Low Risk: Keep maintaining your healthy lifestyle."

    return {
        "cardio_prediction": "Positive" if int(prediction) == 1 else "Negative",
        "probability_percentage": f"{probability:.2f}%",
        "status": message
    }
    @app.post("/predict")
def predict(data: PatientData):
    # 1. Preprocess raw data
    age_bin = get_age_bin(data.age)
    bmi_class = get_bmi_class(data.weight, data.height)
    map_class = get_map_class(data.ap_hi, data.ap_lo)
    
    # 2. Arrange features into a DataFrame with column names

    feature_names = ["gender", "age_bin", "BMI_Class", "MAP_Class", "cholesterol", "gluc", "smoke", "active"]
    features_df = pd.DataFrame([[
        data.gender, age_bin, bmi_class, map_class, 
        data.cholesterol, data.gluc, data.smoke, data.active
    ]], columns=feature_names)

    # 3. Predict
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1] * 100