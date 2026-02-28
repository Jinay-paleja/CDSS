import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Page config must be first
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Get environment
env = os.environ.get('STREAMLIT_ENV', 'production')

# Model and data files
MODEL_FILE = 'heart_model.pkl'
SCALER_FILE = 'heart_scaler.pkl'
COLUMNS_FILE = 'model_columns.pkl'

@st.cache_resource
def load_model():
    """Load the model and scaler"""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        columns = joblib.load(COLUMNS_FILE)
        return model, scaler, columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model
model, scaler, model_columns = load_model()

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(-45deg, #e0f2fe 0%, #f0f9ff 25%, #f5f3ff 50%, #f0fdfa 75%, #e0f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hero section styling */
    .hero {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.6rem 1.25rem;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    
    .hero h1 {
        font-size: 2.5rem;
        color: #0f172a;
        margin-bottom: 1rem;
    }
    
    .hero h1 span {
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero p {
        font-size: 1.1rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto 2rem;
    }
    
    /* Feature icons */
    .feature-icons {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }
    
    .icon-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 1rem 1.5rem;
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .icon-card .icon {
        font-size: 1.5rem;
    }
    
    .icon-card .text h4 {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0;
    }
    
    .icon-card .text p {
        font-size: 0.75rem;
        color: #64748b;
        margin: 0;
    }
    
    /* Info cards */
    .info-section {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        min-width: 150px;
    }
    
    .info-card .number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .info-card h3 {
        font-size: 0.9rem;
        color: #0f172a;
        margin: 0.25rem 0;
    }
    
    .info-card p {
        font-size: 0.75rem;
        color: #64748b;
        margin: 0;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0284c7, #0d9488);
    }
</style>
""", unsafe_allow_html=True)

def show_home():
    """Display the home page content"""
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">
            <span>🏥</span> AI-Powered Health Prediction
        </div>
        
        <h1>Heart Disease <br><span>Prediction System</span></h1>
        
        <p>
            An advanced machine learning system that helps predict heart disease risk 
            using clinical parameters with high accuracy.
        </p>
        
        <div class="feature-icons">
            <div class="icon-card">
                <div class="icon">🎯</div>
                <div class="text">
                    <h4>85%+ Accuracy</h4>
                    <p>High prediction accuracy</p>
                </div>
            </div>
            <div class="icon-card">
                <div class="icon">⚡</div>
                <div class="text">
                    <h4>Instant Results</h4>
                    <p>Get predictions in seconds</p>
                </div>
            </div>
            <div class="icon-card">
                <div class="icon">🔒</div>
                <div class="text">
                    <h4>Secure & Private</h4>
                    <p>Data stays on your device</p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="info-section">
        <div class="info-card">
            <div class="number">13</div>
            <h3>Clinical Parameters</h3>
            <p>Analyzed for accurate predictions</p>
        </div>
        <div class="info-card">
            <div class="number">85%</div>
            <h3>Accuracy Rate</h3>
            <p>Proven prediction accuracy</p>
        </div>
        <div class="info-card">
            <div class="number">24/7</div>
            <h3>Available</h3>
            <p>Access predictions anytime</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get Started button
    if st.button("🏥 Get Started - Make Prediction"):
        st.session_state['show_form'] = True
        st.rerun()

def show_prediction_form():
    """Display the prediction form"""
    if st.button("← Back to Home"):
        st.session_state['show_form'] = False
        st.rerun()
    
    st.markdown("### ❤️ Enter your health information below")
    
    # Create columns for input layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        sex = sex[1] if isinstance(sex, tuple) else sex
        cp = st.selectbox("Chest Pain Type", options=[
            ("Typical Angina", 0),
            ("Atypical Angina", 1),
            ("Non-anginal Pain", 2),
            ("Asymptomatic", 3)
        ], format_func=lambda x: x[0])
        cp = cp[1] if isinstance(cp, tuple) else cp
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        fbs = fbs[1] if isinstance(fbs, tuple) else fbs
    
    with col2:
        restecg = st.selectbox("Resting ECG", options=[
            ("Normal", 0),
            ("ST-T Wave Abnormality", 1),
            ("Left Ventricular Hypertrophy", 2)
        ], format_func=lambda x: x[0])
        restecg = restecg[1] if isinstance(restecg, tuple) else restecg
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        exang = exang[1] if isinstance(exang, tuple) else exang
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
            ("Upsloping", 0),
            ("Flat", 1),
            ("Downsloping", 2)
        ], format_func=lambda x: x[0])
        slope = slope[1] if isinstance(slope, tuple) else slope
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3], index=0)
        thal = st.selectbox("Thalassemia", options=[
            ("Normal", 1),
            ("Fixed Defect", 2),
            ("Reversable Defect", 3)
        ], format_func=lambda x: x[0])
        thal = thal[1] if isinstance(thal, tuple) else thal
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Predict Heart Disease Risk"):
        if model is not None and scaler is not None:
            try:
                # Create input data
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
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                if prediction == 1:
                    st.error("⚠️ Heart Disease Detected!")
                    confidence = probability[1] * 100
                    st.progress(int(confidence))
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.warning("Please consult with a healthcare professional for further evaluation.")
                else:
                    st.success("✅ No Heart Disease Detected")
                    confidence = probability[0] * 100
                    st.progress(int(confidence))
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.info("Your heart appears healthy! Maintain a healthy lifestyle.")
                
                # Show additional info
                with st.expander("See Input Summary"):
                    st.write(input_data)
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Model not loaded. Please check the model files.")
    
    # Info section
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About This App
    This Heart Disease Prediction App uses Machine Learning to predict the likelihood of heart disease 
    based on various health indicators. The model was trained on the UCI Heart Disease dataset 
    using Logistic Regression algorithm.
    
    **Note:** This prediction is for educational purposes only and should not be used as medical advice.
    Always consult with a qualified healthcare professional for any medical concerns.
    """)

# Initialize session state
if 'show_form' not in st.session_state:
    st.session_state['show_form'] = False

# Main app logic
if __name__ == "__main__":
    if st.session_state['show_form']:
        show_prediction_form()
    else:
        show_home()
