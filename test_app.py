import requests
import json

# Test data - sample patient data
test_data = {
    'age': '55',
    'sex': '1',
    'cp': '2',
    'trestbps': '130',
    'chol': '250',
    'fbs': '0',
    'restecg': '1',
    'thalach': '150',
    'exang': '0',
    'oldpeak': '1.5',
    'slope': '2',
    'ca': '0',
    'thal': '2'
}

print("Testing the AI-based CDSS Webapp...")
print("=" * 50)

# Test 1: Check if home page loads
print("\n[Test 1] Checking home page...")
try:
    response = requests.get('http://127.0.0.1:5000/')
    if response.status_code == 200:
        print("✓ Home page loaded successfully!")
    else:
        print(f"✗ Home page returned status code: {response.status_code}")
except Exception as e:
    print(f"✗ Could not connect to home page: {e}")

# Test 2: Make a prediction
print("\n[Test 2] Making a prediction...")
try:
    response = requests.post('http://127.0.0.1:5000/predict', data=test_data)
    if response.status_code == 200:
        print("✓ Prediction request successful!")
        
        # Check if result is in the response
        if 'Heart Disease' in response.text or 'result' in response.text.lower():
            print("✓ Prediction result found in response!")
        else:
            print("Response content:", response.text[:500])
    else:
        print(f"✗ Prediction request returned status code: {response.status_code}")
except Exception as e:
    print(f"✗ Prediction request failed: {e}")

print("\n" + "=" * 50)
print("Testing complete!")
