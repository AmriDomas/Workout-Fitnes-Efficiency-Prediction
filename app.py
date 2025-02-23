import pickle
import pandas as pd
from flask import Flask, request, render_template
from preprocess import preprocess_data
from sklearn.preprocessing import LabelEncoder

# Load model yang sudah dilatih
with open("stacking_best.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load Label Encoders
with open("label_encoders.pkl", "rb") as encoder_file:
    label_encoders = pickle.load(encoder_file)

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
            # Ambil input dari form dengan menangani error input
            data = {
                "Age": int(request.form.get("age", 0)),
                "Gender": request.form.get("gender", "Other"),
                "Height (cm)": int(request.form.get("height", 0)),
                "Weight (kg)": int(request.form.get("weight", 0)),
                "Workout Type": request.form.get("workout_type", "HIIT"),
                "Workout Duration (mins)": int(request.form.get("workout_duration", 0)),
                "Calories Burned": int(request.form.get("calories_burned", 0)),
                "Heart Rate (bpm)": int(request.form.get("heart_rate", 0)),
                "Steps Taken": int(request.form.get("steps_taken", 0)),
                "Distance (km)": float(request.form.get("distance", 0.0)),
                "Workout Intensity": request.form.get("workout_intensity", "Medium"),
                "Sleep Hours": float(request.form.get("sleep_hours", 0.0)),
                "Daily Calories Intake": int(request.form.get("daily_calories_intake", 0)),
                "Resting Heart Rate (bpm)": int(request.form.get("resting_heart_rate", 0)),
                "Mood Before Workout": request.form.get("mood_before_workout", "Neutral"),
                "Mood After Workout": request.form.get("mood_after_workout", "Neutral"),
            }

            df = pd.DataFrame([data])

            # ✅ **Label Encoding untuk data baru**
            categorical_cols = ["Gender", "Workout Type", "Workout Intensity", "Mood Before Workout", "Mood After Workout"]
            
            label_encoders = {}

            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])  # Apply encoding
                label_encoders[col] = le

            # ✅ **Pastikan fitur sesuai dengan format saat training**
            expected_features = model.feature_names_in_
            missing_features = [col for col in expected_features if col not in df.columns]
            extra_features = [col for col in df.columns if col not in expected_features]

            # Tambahkan fitur yang hilang dengan nilai default 0
            for col in missing_features:
                df[col] = 0

            # Hapus fitur yang tidak dikenali oleh model
            df = df[expected_features]

            # ✅ **Prediksi menggunakan model**
            prediction = model.predict(df)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
