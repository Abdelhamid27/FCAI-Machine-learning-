from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback

app = FastAPI()

# 1. تحميل الموديل ومعرفة ترتيب الأعمدة المتوقع
try:
    model = joblib.load('heart_model.pkl')
    # محاولة سحب ترتيب الأعمدة من الموديل أوتوماتيكياً
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_.tolist()
    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_names_in_'):
        expected_features = model.best_estimator_.feature_names_in_.tolist()
    else:
        # الترتيب الافتراضي في حال عدم وجوده داخل الموديل
        expected_features = ["Cluster", "gender", "age_bin", "BMI_Class", "MAP_Class", "cholesterol", "gluc", "smoke", "active"]
    
    print(f"✅ Model loaded. Expected order: {expected_features}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    expected_features = []

class PatientData(BaseModel):
    gender: int
    age: int      # العمر بالسنين
    height: int   # الطول بالسنتيمتر
    weight: float # الوزن بالكيلوجرام
    ap_hi: int    # الضغط الانقباضي
    ap_lo: int    # الضغط الانبساطي
    cholesterol: int
    gluc: int
    smoke: int
    active: int

# دالة تقسيم العمر
def get_age_bin(age):
    if age < 35: return 0
    elif age < 40: return 1
    elif age < 45: return 2
    elif age < 50: return 3
    elif age < 55: return 4
    elif age < 60: return 5
    else: return 6

# دالة تقسيم كتلة الجسم
def get_bmi_class(weight, height_cm):
    bmi = weight / ((height_cm / 100) ** 2)
    if bmi < 18.5: return 1
    elif bmi < 25: return 2
    elif bmi < 30: return 3
    elif bmi < 35: return 4
    elif bmi < 40: return 5
    else: return 6

# دالة تقسيم ضغط الدم
def get_map_class(hi, lo):
    map_val = (hi + 2 * lo) / 3
    if map_val < 80: return 1
    elif map_val < 90: return 2
    elif map_val < 100: return 3
    elif map_val < 110: return 4
    elif map_val < 120: return 5
    else: return 6

@app.post("/predict")
def predict(data: PatientData):
    try:
        # 1. تحويل البيانات لتقسيمات (Bins)
        a_bin = get_age_bin(data.age)
        b_class = get_bmi_class(data.weight, data.height)
        m_class = get_map_class(data.ap_hi, data.ap_lo)
        
        # 2. قواميس ترجمة الأرقام لنصوص وصفية
        age_labels = {0: "Under 35 years", 1: "35-40 years", 2: "40-45 years", 3: "45-50 years", 4: "50-55 years", 5: "55-60 years", 6: "Over 60 years"}
        bmi_labels = {1: "Underweight", 2: "Normal weight", 3: "Overweight", 4: "Obesity Class I", 5: "Obesity Class II", 6: "Extreme Obesity"}
        map_labels = {1: "Low Blood Pressure", 2: "Normal Blood Pressure", 3: "Pre-hypertension", 4: "Stage 1 Hypertension", 5: "Stage 2 Hypertension", 6: "Hypertensive Crisis"}

        # 3. تجميع كل القيم الممكنة
        input_values = {
            "gender": data.gender,
            "age_bin": a_bin,
            "BMI_Class": b_class,
            "MAP_Class": m_class,
            "cholesterol": data.cholesterol,
            "gluc": data.gluc,
            "smoke": data.smoke,
            "active": data.active,
            "Cluster": 0 
        }

        # 4. إعادة ترتيب الأعمدة حسب طلب الموديل
        ordered_data = [[input_values[feat] for feat in expected_features]]
        features_df = pd.DataFrame(ordered_data, columns=expected_features)

        # 5. التوقع
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1] * 100

        # 6. الرد النهائي بالتقرير الطبي
        return {
            "cardio_prediction": "Positive" if int(prediction) == 1 else "Negative",
            "probability_percentage": f"{probability:.2f}%",
            "risk_status": "High Risk" if probability > 50 else "Low Risk",
            "medical_analysis": {
                "age_group": age_labels.get(a_bin, "Unknown"),
                "weight_status": bmi_labels.get(b_class, "Unknown"),
                "blood_pressure_status": map_labels.get(m_class, "Unknown")
            },
            "recommendation": "High probability of heart disease. Please consult a specialist." if probability > 50 else "Low probability. Continue maintaining a healthy lifestyle."
        }

    except Exception as e:
        print("🔥 Traceback error:")
        print(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)