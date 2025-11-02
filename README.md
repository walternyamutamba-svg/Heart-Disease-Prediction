# ❤️ Heart Disease Prediction App

A fully deployed **Machine Learning Web Application** built with **Streamlit**, capable of predicting the likelihood of heart disease based on medical inputs. The app uses a trained **Random Forest model**, applies proper data scaling, features a modern UI with dark mode, supports **PDF report generation**, and logs all predictions for later analysis.

---

## 🚀 Features

✅ **Clean, modern Streamlit UI** (custom CSS + dark mode)
✅ **Sidebar input form** for patient medical data
✅ **Random Forest model** for predicting heart disease
✅ **Data scaling** using a saved StandardScaler
✅ **Detailed prediction output** with probability score
✅ **Downloadable PDF report** (auto-generated)
✅ **Automatic logging** of predictions to CSV
✅ **Easily deployable on Streamlit Cloud**

---

## 🧠 Machine Learning Model

The model was trained using a Random Forest classifier on a heart disease dataset. Features include:

* age
* sex
* cp
* trestbps
* chol
* fbs
* restecg
* thalach
* exang
* oldpeak
* slope
* ca
* thal

The input features must match this order for accurate predictions.

Both the **model** and the **scaler** must be placed in the project directory:

```
random_forest_heart_disease_model.joblib
scaler.joblib
```

---

## 📁 Project Structure

```
heart_disease_app/
│── app.py
│── requirements.txt
│── random_forest_heart_disease_model.joblib
│── scaler.joblib
│── prediction_logs.csv  (auto-created)
│── README.md
```

---

## ▶️ Running the App Locally

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2. Run Streamlit**

```
streamlit run app.py
```

The app will open at:

```
http://localhost:8501
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push the entire project to GitHub
2. Visit **[https://share.streamlit.io](https://share.streamlit.io)**
3. Click **Deploy App**
4. Select your GitHub repo
5. Set:

   * **Main file:** `app.py`
   * **Requirements file:** `requirements.txt`
6. Deploy ✅

Your app will be hosted at:

```
https://your-app-name.streamlit.app
```

---

## 📄 PDF Report Generation

Every prediction generates a **downloadable PDF** containing:

* Prediction (Heart Disease / No Heart Disease)
* Probability score
* Input values
* Timestamp

Useful for medical reporting or offline analysis.

---

## 📝 Logging

All predictions are automatically saved into `prediction_logs.csv` with fields:

* All input features
* Predicted class
* Probability
* Timestamp

This allows tracking, auditing, and potential model retraining.

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
reportlab
```

---

## 👤 Author

Developed by **Walter Nyamutamba** — Data Scientist, Analyst, and Machine Learning Engineer.

---

## ⭐ Support the Project

If you found this useful, consider starring the repository on GitHub!

---

## 🐛 Issues

Feel free to open an issue for feature requests, bugs, or enhancements.

---
