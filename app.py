import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from config import config

# Get environment from FLASK_ENV or default to production
env = os.environ.get('FLASK_ENV', 'production')
app = Flask(__name__)
app.config.from_object(config.get(env, config['production']))

# Model and data files
MODEL_FILE = 'heart_model.pkl'
SCALER_FILE = 'heart_scaler.pkl'
COLUMNS_FILE = 'model_columns.pkl'
DATA_FILE = 'heart.csv'

def train_model():
    """Train and save the heart disease prediction model"""
    print("Training model...")
    data = pd.read_csv(DATA_FILE)
    
    # The target column is 'target' in heart.csv
    target_col = "target"
    
    if target_col in data.columns:
        data = data.dropna(subset=[target_col])
    
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = data.select_dtypes(include=["object", "string"]).columns
    
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    for col in categorical_cols:
        if col != target_col:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Separate features and target before one-hot encoding
    y = data[target_col]
    X = data.drop(target_col, axis=1)
    
    # One-hot encode only the features
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(list(X.columns), COLUMNS_FILE)
    
    print("Model saved successfully!")
    return model, scaler, list(X.columns)

def load_model():
    """Load the model or train if not exists"""
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(COLUMNS_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        columns = joblib.load(COLUMNS_FILE)
        print("Model loaded successfully!")
    else:
        print("Model not found. Training new model...")
        model, scaler, columns = train_model()
    return model, scaler, columns

# Load model at startup
model, scaler, model_columns = load_model()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('home.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    """Render the prediction form page"""
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    """Make prediction based on user input"""
    try:
        # Get input values from form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Create input dataframe
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Create dataframe with all columns
        input_df = pd.DataFrame([input_data])
        
        # Add missing columns with 0
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[model_columns]
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        confidence = probability[prediction] * 100
        
        # Store data for display
        patient_data = input_data
        
        return render_template('result.html', 
                             result=result_text, 
                             confidence=f"{confidence:.2f}",
                             prediction=prediction,
                             patient_data=patient_data)
    
    except Exception as e:
        return render_template('predict.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
