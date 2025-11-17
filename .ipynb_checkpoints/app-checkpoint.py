import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_energy_model.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_energy_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_hourly.joblib")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")

FEATURE_COLS = [
    'Global_reactive_power','Voltage','Global_intensity',
    'Sub_metering_1','Sub_metering_2','Sub_metering_3',
    'hour','day_of_week','is_weekend',
    'roll_3','roll_6','roll_12','lag_1','lag_2','lag_3'
]

st.set_page_config(page_title="Energy Forecast", layout="wide")
st.title("Smart Energy Consumption Forecasting")
st.markdown("**Modes:** CSV Upload (recommended) or Manual Input. LSTM requires CSV (sequence) mode.")

@st.cache_data
def load_models():
    models = {}
    # RF
    if os.path.exists(RF_MODEL_PATH):
        models['rf'] = joblib.load(RF_MODEL_PATH)
    # XGB
    if os.path.exists(XGB_MODEL_PATH):
        models['xgb'] = joblib.load(XGB_MODEL_PATH)
    # Scaler
    if os.path.exists(SCALER_PATH):
        models['scaler'] = joblib.load(SCALER_PATH)
    # LSTM
    if os.path.exists(LSTM_MODEL_PATH):
        try:
            models['lstm'] = load_model(LSTM_MODEL_PATH)
        except Exception as e:
            st.warning(f"Error loading LSTM model: {e}")
    return models

def basic_preprocess(df):
    df = df.copy()
    if 'datetime' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        else:
            raise ValueError("No 'datetime' column found in uploaded CSV.")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    for c in df.columns:
        if c != 'datetime':
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                pass
    # time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    return df

def add_hourly_rolling_lags(df):
    df = df.copy()
    df['datetime_hour'] = df['datetime'].dt.floor('H')
    hourly = df.set_index('datetime').resample('H').mean()
    hourly['roll_3'] = hourly['Global_active_power'].rolling(window=3, min_periods=1).mean()
    hourly['roll_6'] = hourly['Global_active_power'].rolling(window=6, min_periods=1).mean()
    hourly['roll_12'] = hourly['Global_active_power'].rolling(window=12, min_periods=1).mean()
    hourly['lag_1'] = hourly['Global_active_power'].shift(1)
    hourly['lag_2'] = hourly['Global_active_power'].shift(2)
    hourly['lag_3'] = hourly['Global_active_power'].shift(3)
    hourly[['roll_3','roll_6','roll_12','lag_1','lag_2','lag_3']] = hourly[['roll_3','roll_6','roll_12','lag_1','lag_2','lag_3']].bfill()
    feats = hourly[['roll_3','roll_6','roll_12','lag_1','lag_2','lag_3']].rename(columns={
        'roll_3':'roll_3','roll_6':'roll_6','roll_12':'roll_12','lag_1':'lag_1','lag_2':'lag_2','lag_3':'lag_3'
    })
    df = df.merge(feats, left_on='datetime_hour', right_index=True, how='left')
    df[['roll_3','roll_6','roll_12','lag_1','lag_2','lag_3']] = df[['roll_3','roll_6','roll_12','lag_1','lag_2','lag_3']].bfill()
    return df

def make_feature_matrix(df):
    df2 = add_hourly_rolling_lags(df)
    missing = [c for c in FEATURE_COLS if c not in df2.columns]
    if missing:
        raise ValueError(f"Missing required features after preprocessing: {missing}")
    X = df2[FEATURE_COLS].copy()
    y = None
    if 'Global_active_power' in df2.columns:
        y = df2['Global_active_power'].copy()
    return df2, X, y

def predict_with_model(model, X):
    return model.predict(X)
    
models = load_models()
st.sidebar.header("Options")
mode = st.sidebar.radio("Mode", ("CSV Upload", "Manual Input"))

st.sidebar.markdown("**Models available:**")
st.sidebar.write(", ".join(models.keys()) if models else "No pre-saved models found (you can upload CSV and train).")

if mode == "CSV Upload":
    st.header("CSV Upload Mode")
    uploaded = st.file_uploader("Upload cleaned CSV (with datetime & active power). Example: cleaned_preprocessed_energy_data.csv", type=['csv'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("File loaded. Running preprocessing...")
            df = basic_preprocess(df)
            st.write("Preview of data:")
            st.dataframe(df.head(6))
            # build features
            df_feat, X_all, y_all = make_feature_matrix(df)
            st.write("Features built. Sample:")
            st.dataframe(df_feat[['datetime','Global_active_power'] + FEATURE_COLS].head(6))
            # selection: choose model
            st.subheader("Prediction options")
            model_choice = st.selectbox("Choose model", options=["RandomForest","XGBoost","LSTM","All (compare)"])
            predict_n = st.number_input("Predict next how many rows? (for CSV mode we predict on the available rows)", min_value=1, value=1, step=1)
            
            if not models:
                st.warning("No saved models found. You can train quick RF/XGB on uploaded CSV (this will run on your machine).")
                if st.button("Train quick RF & XGB now"):
                    # quick train on first 80% (time-based)
                    split_idx = int(len(X_all) * 0.8)
                    Xtr, Xte = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
                    ytr, yte = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
                    with st.spinner("Training RandomForest..."):
                        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        rf.fit(Xtr, ytr)
                        joblib.dump(rf, RF_MODEL_PATH)
                        models['rf'] = rf
                    with st.spinner("Training XGBoost..."):
                        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
                        xgb.fit(Xtr, ytr)
                        joblib.dump(xgb, XGB_MODEL_PATH)
                        models['xgb'] = xgb
                    st.success("Trained and saved RF and XGB models to models/ folder.")
           
            if st.button("Run Predictions"):
                results = {}
                split_idx = int(len(X_all) * 0.8)
                X_test_local = X_all.iloc[split_idx:].copy()
                y_test_local = y_all.iloc[split_idx:].copy() if y_all is not None else None
                st.write(f"Predicting on {X_test_local.shape[0]} rows (chronological test split).")
                if model_choice in ("RandomForest","All (compare)"):
                    if 'rf' in models:
                        preds_rf = predict_with_model(models['rf'], X_test_local)
                        results['rf'] = preds_rf
                        st.write("RandomForest predictions (sample):")
                        st.line_chart(pd.DataFrame({"actual": y_test_local.values[:200], "predicted_rf": preds_rf[:200]}))
                    else:
                        st.error("RandomForest model not found.")
                if model_choice in ("XGBoost","All (compare)"):
                    if 'xgb' in models:
                        preds_xgb = predict_with_model(models['xgb'], X_test_local)
                        results['xgb'] = preds_xgb
                        st.write("XGBoost predictions (sample):")
                        st.line_chart(pd.DataFrame({"actual": y_test_local.values[:200], "predicted_xgb": preds_xgb[:200]}))
                    else:
                        st.error("XGBoost model not found.")
                if model_choice in ("LSTM","All (compare)"):
                    if 'lstm' in models and 'scaler' in models:
                        st.info("Preparing sequences for LSTM (hourly). LSTM requires hourly sequence data.")
                        hourly = df.set_index('datetime').resample('H').mean()
                        hourly['hour'] = hourly.index.hour
                        hourly['day_of_week'] = hourly.index.dayofweek
                        hourly['is_weekend'] = hourly['day_of_week'].isin([5,6]).astype(int)
                        lstm_features = ['Global_active_power','roll_3h','roll_6h','roll_12h','lag_1h','lag_2h','lag_3h','hour','day_of_week','is_weekend']
                        if 'roll_3h' not in hourly.columns:
                            hourly['roll_3h'] = hourly['Global_active_power'].rolling(3, min_periods=1).mean()
                            hourly['roll_6h'] = hourly['Global_active_power'].rolling(6, min_periods=1).mean()
                            hourly['roll_12h'] = hourly['Global_active_power'].rolling(12, min_periods=1).mean()
                            hourly['lag_1h'] = hourly['Global_active_power'].shift(1)
                            hourly['lag_2h'] = hourly['Global_active_power'].shift(2)
                            hourly['lag_3h'] = hourly['Global_active_power'].shift(3)
                            hourly = hourly.bfill()
                       
                        scaler = models['scaler']
                        sub = hourly[lstm_features].dropna()
                        scaled = scaler.transform(sub.values)
                        SEQ_LEN = 24
                        Xs = []
                        for i in range(len(scaled) - SEQ_LEN):
                            Xs.append(scaled[i:i+SEQ_LEN])
                        Xs = np.array(Xs)
                        preds_lstm_scaled = models['lstm'].predict(Xs, verbose=0)
                        
                        zeros = np.zeros((preds_lstm_scaled.shape[0], scaled.shape[1]-1))
                        pred_full = np.hstack([preds_lstm_scaled, zeros])
                        pred_inv = scaler.inverse_transform(pred_full)[:,0]
                        st.line_chart(pd.DataFrame({"lstm_pred": pred_inv[:200]}))
                    else:
                        st.error("LSTM or scaler not found in models/. Please save them or use CSV mode training.")
                st.success("Predictions complete.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

else:
    st.header("Manual Input Mode (single-row prediction)")
    st.markdown("Enter feature values. LSTM is not supported for single manual rows; use CSV mode for sequence-based predictions.")
    # create manual input widgets based on FEATURE_COLS
    manual_vals = {}
    st.subheader("Energy feature inputs")
    # show a few key inputs (you can add more or use defaults)
    manual_vals['Global_reactive_power'] = st.number_input("Global_reactive_power", value=0.1, format="%.6f")
    manual_vals['Voltage'] = st.number_input("Voltage", value=234.0, format="%.3f")
    manual_vals['Global_intensity'] = st.number_input("Global_intensity", value=10.0, format="%.3f")
    manual_vals['Sub_metering_1'] = st.number_input("Sub_metering_1", value=0.0)
    manual_vals['Sub_metering_2'] = st.number_input("Sub_metering_2", value=0.0)
    manual_vals['Sub_metering_3'] = st.number_input("Sub_metering_3", value=0.0)
    manual_vals['hour'] = st.slider("Hour", 0, 23, 12)
    manual_vals['day_of_week'] = st.slider("Day of week (0=Mon)", 0, 6, 2)
    manual_vals['is_weekend'] = 1 if manual_vals['day_of_week'] in [5,6] else 0
    # rolling & lag values (we allow manual entry or defaults)
    manual_vals['roll_3'] = st.number_input("roll_3 (3-hour avg of global active)", value=1.0, format="%.6f")
    manual_vals['roll_6'] = st.number_input("roll_6 (6-hour avg)", value=1.0, format="%.6f")
    manual_vals['roll_12'] = st.number_input("roll_12 (12-hour avg)", value=1.0, format="%.6f")
    manual_vals['lag_1'] = st.number_input("lag_1 (previous hour)", value=1.0, format="%.6f")
    manual_vals['lag_2'] = st.number_input("lag_2 (2 hours ago)", value=1.0, format="%.6f")
    manual_vals['lag_3'] = st.number_input("lag_3 (3 hours ago)", value=1.0, format="%.6f")

    chosen_model = st.selectbox("Choose model for manual prediction", options=["RandomForest","XGBoost"])
    if st.button("Predict (manual)"):
        row = pd.DataFrame([manual_vals])
        # ensure column order
        X_manual = row[FEATURE_COLS]
        if chosen_model == "RandomForest":
            if 'rf' in models:
                pred = models['rf'].predict(X_manual)[0]
                st.success(f"Predicted Global_active_power (RF): {pred:.6f}")
            else:
                st.error("RandomForest model not found in models/ folder.")
        elif chosen_model == "XGBoost":
            if 'xgb' in models:
                pred = models['xgb'].predict(X_manual)[0]
                st.success(f"Predicted Global_active_power (XGB): {pred:.6f}")
            else:
                st.error("XGBoost model not found in models/ folder.")

st.markdown("---")
st.markdown("**Notes:** LSTM predictions require a saved scaler and LSTM model (see README). Models trained in the notebook should be saved into the `models/` directory.")
