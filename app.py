import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

st.title("📈 Store Sales Forecasting App")
st.markdown("Forecasting next 30 days of sales using XGBoost")

uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['Order Date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['Sales'] = df['Sales'].fillna(method='ffill')
    df = df.dropna(subset=['Sales'])

    def create_features(data):
        data['lag_1'] = data['Sales'].shift(1)
        data['lag_2'] = data['Sales'].shift(2)
        data['lag_7'] = data['Sales'].shift(7)
        data['roll_mean_3'] = data['Sales'].shift(1).rolling(window=3).mean()
        data['roll_std_3'] = data['Sales'].shift(1).rolling(window=3).std()
        data['roll_mean_7'] = data['Sales'].shift(1).rolling(window=7).mean()
        data['roll_std_7'] = data['Sales'].shift(1).rolling(window=7).std()
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        data['day_of_week'] = data['date'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        return data

    df = create_features(df)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    feature_cols = [
        'lag_1', 'lag_2', 'lag_7',
        'roll_mean_3', 'roll_std_3', 'roll_mean_7', 'roll_std_7',
        'month', 'year', 'day_of_week', 'is_weekend'
    ]

    X_train = df[feature_cols].astype(np.float32)
    y_train = df['Sales'].astype(np.float32)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    last_date = df['date'].max()
    future_days = 30
    df_fc = df.copy()
    future_preds = []

    for _ in range(future_days):
        next_date = last_date + pd.Timedelta(days=1)
        new_row = {'date': next_date, 'Sales': np.nan}
        df_fc = pd.concat([df_fc, pd.DataFrame([new_row])], ignore_index=True)
        df_fc = create_features(df_fc)
        df_fc.fillna(df_fc.mean(numeric_only=True), inplace=True)
        X_pred = df_fc[feature_cols].iloc[[-1]].astype(np.float32)
        next_sales = model.predict(X_pred)[0]
        df_fc.at[df_fc.index[-1], 'Sales'] = next_sales
        future_preds.append({'date': next_date, 'Sales': next_sales})
        last_date = next_date

    forecast_df = pd.DataFrame(future_preds)

    st.subheader("📊 Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['date'], df['Sales'], label='Historical Sales', color='blue')
    ax.plot(forecast_df['date'], forecast_df['Sales'], label='Forecast', color='orange', linestyle='--')
    ax.set_title("30-Day Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("📅 Forecast Table")
    st.dataframe(forecast_df)
