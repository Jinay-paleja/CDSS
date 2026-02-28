# HeartCare AI - Clinical Decision Support System

An AI-powered web application for heart disease prediction using machine learning.

![HeartCare AI](https://img.shields.io/badge/HeartCare-AI-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![Flask](https://img.shields.io/badge/Flask-3.1.3-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **AI-Powered Predictions**: Uses Logistic Regression model with 85%+ accuracy
- **Professional UI**: Modern, responsive design with gradient aesthetics
- **Multi-Page Interface**: Home, About, Predict, and Result pages
- **Instant Results**: Real-time heart disease risk assessment
- **Confidence Scores**: Shows prediction confidence percentage
- **Secure Processing**: All predictions processed locally

## 🏗️ Tech Stack

- **Backend**: Python 3.12, Flask 3.1
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Deployment**: Gunicorn, Render/Fly.io/PythonAnywhere
- **Frontend**: HTML5, CSS3, Modern JavaScript

## 📋 Clinical Parameters

The system analyzes 13 clinical parameters:
- Age, Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Serum Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG Results (restecg)
- Maximum Heart Rate (thalach)
- Exercise Induced Angina (exang)
- ST Depression (oldpeak)
- Slope of Peak Exercise ST (slope)
- Number of Major Vessels (ca)
- Thalassemia Type (thal)

## 🚀 Quick Start

### Local Development

```
bash
# Clone the repository
git clone <your-repo-url>
cd HeartCare-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## ☁️ Deployment

### Option 1: Render (Recommended - Free)

1. **Push to GitHub**
   
```
bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/HeartCare-AI.git
   git push -u origin main
   
```

2. **Deploy on Render**
   - Go to [render.com](https://render.com) and sign up
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Use these settings:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`
   - Click "Create Web Service"

3. **Environment Variables**
   - In Render dashboard, add: `FLASK_ENV = production`

Your app will be live at `https://your-app-name.onrender.com`

### Option 2: Fly.io

```
bash
# Install flyctl
brew install flyctl  # macOS
# or
winget install flyctl.io/flyctl

# Login
flyctl auth login

# Launch
flyctl launch

# Set environment
flyctl secrets set FLASK_ENV=production

# Deploy
flyctl deploy
```

### Option 3: PythonAnywhere

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Go to "Files" → Upload files
3. Open a Bash console and run:
   
```bash
   pip install -r requirements.txt
   
```
4. Go to "Web" → Add a new web app
5. Configure WSGI file to point to your app

### Option 4: Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Add the required environment variables
4. Deploy!

## 📁 Project Structure

```
HeartCare-AI/
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment config (Render)
├── runtime.txt           # Python runtime version
├── .gitignore            # Git ignore rules
├── heart.csv             # Training dataset
├── heart_model.pkl       # Trained model
├── heart_scaler.pkl      # Data scaler
├── model_columns.pkl     # Feature columns
├── templates/
│   ├── home.html         # Home page
│   ├── about.html        # About page
│   ├── predict.html     # Prediction form
│   └── result.html      # Result page
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment | `production` |
| `SECRET_KEY` | App secret key | Auto-generated |
| `PORT` | Server port | `5000` |

### Production Settings

For production deployment, update `app.py`:

```
python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
```

## ⚠️ Disclaimer

This application is for educational and screening purposes only. The predictions made by this system should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical evaluation.

## 📄 License

MIT License - Feel free to use this project for educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

Made with ❤️ for Healthcare AI
