import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import chardet
import io

# --- Title ---
st.title("📈 Store Sales Forecasting App")
st.markdown("Upload a sales CSV file. Forecasting the next 30 days using XGBoost.")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📂 Upload Sales CSV", type=["csv"])

if uploaded_file:
    try:
        # --- Detect file encoding ---
        raw_bytes = uploaded_file.read()
        result = chardet.detect(raw_bytes)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'

        # --- Decode and read CSV ---
        decoded_str = raw_bytes.decode(encoding)
        df = pd.read_csv(io.StringIO(decoded_str))
        df.columns = df.columns.str.strip().str.lower()  # Normalize column names

        # --- Validate required columns ---
        required_columns = ['order date', 'sales']
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ Required columns missing. Expected: {required_columns}")
            st.stop()

        # --- Preprocessing ---
        df['date'] = pd.to_datetime(df['order date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df = df.dropna(subset=['date', 'sales'])
        df = df.sort_values('date').reset_index(drop=True)

        if len(df) < 10:
            st.warning("⚠️ Not enough data for forecasting. Please upload a larger dataset.")
            st.stop()

        # --- Feature Engineering ---
        def create_features(data):
            data['lag_1'] = data['sales'].shift(1)
            data['lag_2'] = data['sales'].shift(2)
            data['lag_7'] = data['sales'].shift(7)
            data['roll_mean_3'] = data['sales'].shift(1).rolling(window=3).mean()
            data['roll_std_3'] = data['sales'].shift(1).rolling(window=3).std()
            data['roll_mean_7'] = data['sales'].shift(1).rolling(window=7).mean()
            data['roll_std_7'] = data['sales'].shift(1).rolling(window=7).std()
            data['month'] = data['date'].dt.month
            data['year'] = data['date'].dt.year
            data['day_of_week'] = data['date'].dt.dayofweek
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            return data

        df = create_features(df)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # --- Model Training ---
        feature_cols = [
            'lag_1', 'lag_2', 'lag_7',
            'roll_mean_3', 'roll_std_3',
            'roll_mean_7', 'roll_std_7',
            'month', 'year', 'day_of_week', 'is_weekend'
        ]
        X_train = df[feature_cols].astype(np.float32)
        y_train = df['sales'].astype(np.float32)

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        # --- Forecasting ---
        future_days = 30
        last_date = df['date'].max()
        df_fc = df.copy()
        future_preds = []

        for _ in range(future_days):
            next_date = last_date + pd.Timedelta(days=1)
            new_row = {'date': next_date, 'sales': np.nan}
            df_fc = pd.concat([df_fc, pd.DataFrame([new_row])], ignore_index=True)
            df_fc = create_features(df_fc)
            df_fc.fillna(df_fc.mean(numeric_only=True), inplace=True)
            X_pred = df_fc[feature_cols].iloc[[-1]].astype(np.float32)
            next_sales = model.predict(X_pred)[0]
            df_fc.at[df_fc.index[-1], 'sales'] = next_sales
            future_preds.append({'date': next_date, 'sales': next_sales})
            last_date = next_date

        forecast_df = pd.DataFrame(future_preds)

        # --- Plot ---
        st.subheader("📊 Forecast Plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['date'], df['sales'], label='Historical Sales', color='blue')
        ax.plot(forecast_df['date'], forecast_df['sales'], label='Forecast', color='orange', linestyle='--')
        ax.set_title("30-Day Sales Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # --- Table ---
        st.subheader("📅 Forecast Table")
        st.dataframe(forecast_df)

    except UnicodeDecodeError:
        st.error("❌ Unicode decode error. Try saving your CSV in UTF-8 encoding or let the app auto-detect it.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")
