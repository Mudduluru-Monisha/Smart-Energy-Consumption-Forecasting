# Smart-Energy-Consumption-Forecasting
AI-based project to predict electricity demand using machine learning.

# **Smart Energy Consumption Forecasting Using Machine Learning**

This project predicts future household electricity usage using Machine Learning models. Users can upload their energy dataset and view predicted consumption, compare model performance, and visualize trends through an interactive Streamlit web app.

## **📌 Features**
* Upload CSV data and automatically preprocess it
* Predict electricity usage using:

  * **Random Forest**
  * **XGBoost**
  * **LSTM (sequence-based)**
* Visualize actual vs predicted values
* User-friendly web app built with **Streamlit**
* Rolling averages, lag features, and time-based insights
* Compare multiple models on the same graph

---

## **📁 Folder Structure**

```
Smart Energy Consumption Forecasting/
│
├── app.py                     # Streamlit application
├── models/                    # Pretrained ML models
│     ├── rf_energy_model.joblib
│     ├── xgb_energy_model.joblib
│     ├── lstm_model.h5
│     └── scaler_hourly.joblib
│
├── data/                      # Your datasets (optional)
├── README.md
└── requirements.txt
```

---

## **🛠 Tools & Technologies**

* **Python**
* **Pandas**, **NumPy**
* **Scikit-Learn**
* **XGBoost**
* **TensorFlow/Keras**
* **Streamlit** for web app
* **Joblib** for saving models

---

## **🚀 How to Run**

1. Install required packages:

   ```
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:

   ```
   streamlit run app.py
   ```
3. Open the URL in your browser:

   * Local: `http://localhost:8501`

---

## **📂 Input Data Format**

Your CSV should contain at least:

* `datetime` column
* `Global_active_power`
* Other power-related features (voltage, intensity, sub-metering)

Example:

```
datetime,Global_active_power,Voltage,Global_intensity,Sub_metering_1,...
2007-01-01 00:00:00,1.234,234.56,12.4,0,...
```

---

## **📊 Output**

* Predicted consumption values
* Line charts showing actual vs predicted
* Option to choose which model(s) to apply
* Insights for selected future time steps

---

## **🎯 Project Goal**

To help users understand, analyze, and forecast electricity usage so they can plan their consumption better and reduce unnecessary energy expenditure.

---

## **📌 Future Enhancements**

* Support for day-level and week-level forecasting
* Integration with IoT smart meters
* Appliance-level prediction
* Mobile-friendly UI

