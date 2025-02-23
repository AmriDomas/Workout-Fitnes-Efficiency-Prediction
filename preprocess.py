import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load label encoders
with open("label_encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)


def preprocess_data(df, label_encoders):
    # Hitung kolom tambahan dengan menangani pembagian nol
    df['Margin Calories Burned'] = df['Calories Burned'] - df['Daily Calories Intake']
    df['Margin Heart Rate'] = df['Heart Rate (bpm)'] - df['Resting Heart Rate (bpm)']
    df['Ratio Calories Burned'] = df['Calories Burned'] / df['Daily Calories Intake']
    df['Ratio Heart Rate'] = df['Heart Rate (bpm)'] / df['Resting Heart Rate (bpm)']
    df['Ratio Duration Calories Burned'] = df['Workout Duration (mins)'] / df['Ratio Calories Burned']
    df['Ration WD Sleep Hours'] = df['Workout Duration (mins)'] / df['Sleep Hours']

    # Tangani pembagian dengan nol atau NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Label Encoding untuk kolom kategori
    categorical_cols = ["Gender", "Workout Type", "Workout Intensity", "Mood Before Workout", "Mood After Workout"]

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Apply encoding
        label_encoders[col] = le

    # Hapus kolom yang tidak diperlukan
    df_clean = df.drop(['User ID', 'Water Intake (liters)', 'VO2 Max', 'Body Fat (%)'], axis=1, errors='ignore')

    return df_clean
