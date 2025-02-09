
# Heart Disease Prediction App

## 📌 Overview
This is a **Heart Disease Prediction** web application built using **Streamlit** for the frontend and **Logistic Regression** for the backend model. The dataset is sourced from **Kaggle (Open Source)** and is used to train a machine learning model that predicts whether a person has heart disease based on various health metrics.

## 🔥 Features
- **User Input Form**: Enter health parameters like age, cholesterol, blood pressure, etc.
- **Machine Learning Model**: Uses **Logistic Regression** for prediction.
- **Scalable Data Processing**: Standardizes input features using **StandardScaler**.
- **Interactive Frontend**: Built using **Streamlit** for easy usability.

## 🏗️ Tech Stack
### Backend:
- **Python**
- **Scikit-Learn** (Machine Learning Model - Logistic Regression)
- **Pandas & NumPy** (Data Processing)
- **Joblib** (Model Serialization)

### Frontend:
- **Streamlit** (For creating an interactive UI)

## 🗂 Dataset
The dataset is obtained from **Kaggle** and includes the following features:
- **Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, ECG Results, Max Heart Rate, Exercise Induced Angina, ST Depression, Slope of Peak Exercise ST, Major Vessels, and Thalassemia.**
- **Target Variable:** 1 = Heart Disease, 0 = Healthy

## 🚀 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Heart-disease-prediction.git
   cd Heart-disease-prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Model Training
To retrain the model on updated data:
```bash
python train_model.py
```
This will:
- Load the dataset
- Preprocess the data
- Train a **Logistic Regression** model
- Save the trained model (`heart_disease_model.pkl`) and scaler (`scaler.pkl`)

## 🖥️ Usage
- Enter the required health parameters in the web app.
- Click the **Predict Heart Disease** button.
- The app will display whether heart disease is detected or not.

## 📜 License
This project is open-source and available under the **MIT License**.

## 🤝 Acknowledgments
- **Dataset:** [Kaggle Heart Disease Dataset](https://www.kaggle.com/)
- **Machine Learning Framework:** Scikit-Learn
- **Frontend:** Streamlit

## ⭐ Contributing
If you'd like to contribute, feel free to fork the repository and submit a pull request!

---
Made with ❤️ by [Your Name]

