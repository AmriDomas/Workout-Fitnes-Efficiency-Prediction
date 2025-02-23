# Workout-Fitness-Efficiency-Prediction

## 📌 Project Overview
The **Workout-Fitness-Efficiency-Prediction** project is a machine learning-based web application designed to predict workout efficiency based on various user input parameters. The system utilizes a Stacking Regressor model trained with multiple machine learning algorithms to provide accurate fitness insights.

## 🚀 Features
- Predicts workout efficiency based on user input (age, gender, height, weight, workout type, duration, calories burned, etc.).
- Uses **Stacking Regressor** with optimized hyperparameters.
- Flask-based web application for real-time predictions.
- Supports categorical feature encoding using **LabelEncoder**.
- Includes preprocessing steps to handle missing values and feature engineering.

## 🏗️ Tech Stack
- **Programming Language**: Python
- **Machine Learning**: Scikit-learn (Stacking Regressor, Ridge, Random Forest, Gradient Boosting, AdaBoost)
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Pickle
- **Deployment**: Flask (local server)

## 📂 Project Structure
```
├── app.py                 # Flask application
├── preprocess.py          # Preprocessing functions
├── stacking_best.pkl      # Trained machine learning model
├── label_encoders.pkl     # Saved LabelEncoders for categorical features
├── templates
│   ├── index.html         # Frontend template
├── README.md              # Project documentation
```

## 🏋️‍♂️ How to Use
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/AmriDomas/Workout-Fitness-Efficiency-Prediction.git
cd Workout-Fitness-Efficiency-Prediction
```

### 2️⃣ Install Dependencies
Make sure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
python app.py
```
The application will run locally at `http://127.0.0.1:5000/`.

### 4️⃣ Make Predictions
1. Open the web application in your browser.
2. Enter your fitness details (age, gender, workout type, etc.).
3. Click on **Predict** to get efficiency insights.

## ⚙️ Model Training
The model was trained using:
```python
base_learners = [
    ('gb', gb_best),
    ('ada', ada_best),
    ('rf', rf_best)
]

stacking_regressor = StackingRegressor(estimators=base_learners, final_estimator=Ridge())
```
After tuning, the best model (`stacking_best.pkl`) is saved for deployment.

## ⚠️ Common Errors & Fixes
1. **"The feature names should match those that were passed during fit"**  
   - Ensure that column names in the input match the ones used during training.
2. **"Could not convert string to float: 'Male'"**  
   - Ensure categorical variables are encoded using LabelEncoder before prediction.
3. **"argument of type 'LabelEncoder' is not iterable"**  
   - Use `list(label_encoders[col].classes_)` instead of `set(label_encoders[col].classes_)` when checking for valid categories.

## 📌 Future Enhancements
- Integrate Deep Learning models (e.g., TensorFlow, PyTorch).
- Add more features like VO2 Max, Body Fat %, and more workout intensity levels.
- Deploy using **Docker** or cloud platforms (AWS, GCP, Heroku).

## 🤝 Contributing
Feel free to contribute! Fork the repo, make changes, and submit a pull request.

## 📬 Contact
For any questions or collaborations, reach out at **amrisphotography@gmail.com** or open an issue in the repository.

---
🚀 *Stay fit and improve your workout efficiency with ML-powered insights!*

