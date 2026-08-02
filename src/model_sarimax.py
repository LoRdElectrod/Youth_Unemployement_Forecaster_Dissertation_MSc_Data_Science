import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Suppress statsmodels convergence warnings to keep the console clean
warnings.filterwarnings("ignore")

def run_sarimax_model():
    print("[-] Initializing SARIMAX (ARIMA) Modeling...")
    file_path = "../data/processed/ml_matrix.csv"
    
    if not os.path.exists(file_path):
        print("[!] Error: ml_matrix.csv not found.")
        return
        
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define features to use as external regressors (exog)
    exog_features = ['GDP_Value_mil', 'Inflation_Rate', 'UK_Vacancies_Thousands', 'BoE_Base_Rate']
    target = 'Youth_Unemployment_Rate'
    
    regions = df['Region'].unique()
    
    for region in regions:
        print(f"\n[>] Training SARIMAX Model for: {region}")
        
        # [FIX 1]: Set the Date column as the true DataFrame index for temporal awareness
        region_df = df[df['Region'] == region].copy()
        region_df.set_index('Date', inplace=True)
        
        # Explicitly infer and set the frequency (Quarterly)
        inferred_freq = pd.infer_freq(region_df.index)
        if inferred_freq:
            region_df.index.freq = inferred_freq
        
        # --- TRAIN/TEST SPLIT ---
        train = region_df[region_df.index.year <= 2023].copy()
        test = region_df[region_df.index.year >= 2024].copy()
        
        y_train = train[target]
        exog_train = train[exog_features]
        
        y_test = test[target]
        exog_test = test[exog_features]
        
        # --- STATIONARITY CHECK (ADF TEST) ---
        print("    * Running ADF Test for stationarity...")
        adf_result = adfuller(y_train.dropna())
        p_value = adf_result[1]
        print(f"      - ADF p-value: {p_value:.4f}")
        
        # If p-value < 0.05, data is stationary (d=0). Else, it needs differencing (d=1).
        d_term = 0 if p_value < 0.05 else 1
        print(f"      - Data is {'stationary' if d_term == 0 else 'non-stationary'}. Dynamically setting d={d_term}.")
        
        # --- [FIX 2]: SCALING EXOGENOUS FEATURES ---
        scaler = StandardScaler()
        # Fit on train, transform on train and test
        exog_train_scaled = scaler.fit_transform(exog_train)
        exog_test_scaled = scaler.transform(exog_test)
        
        # Convert back to DataFrame to preserve the DatetimeIndex
        exog_train_scaled = pd.DataFrame(exog_train_scaled, columns=exog_features, index=exog_train.index)
        exog_test_scaled = pd.DataFrame(exog_test_scaled, columns=exog_features, index=exog_test.index)
        
        # --- MODEL CONFIGURATION ---
        model = SARIMAX(
            endog=y_train,
            exog=exog_train_scaled,
            order=(1, d_term, 1),
            seasonal_order=(1, 1, 1, 4),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit the model
        print("    * Fitting model (this may take a moment)...")
        results = model.fit(disp=False)
        
        # --- [FIX 3]: FORECASTING ---
        forecast = results.forecast(steps=len(test), exog=exog_test_scaled)
        
        # --- EVALUATION ---
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        
        print(f"    * SARIMAX MAE: {mae:.2f}%")
        print(f"    * SARIMAX RMSE: {rmse:.2f}%")
        
        # --- VISUALIZATION ---
        plt.figure(figsize=(10, 5))
        plt.plot(train.index, y_train, label='Training Data', color='black')
        plt.plot(test.index, y_test, label='Actual Data (2024-2025)', color='blue', marker='o')
        plt.plot(test.index, forecast, label='SARIMAX Forecast', color='orange', linestyle='--', marker='x')
        
        plt.title(f'SARIMAX Backtest: {region} (2024-2025)')
        plt.xlabel('Date')
        plt.ylabel('Youth Unemployment Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()

if __name__ == "__main__":
    run_sarimax_model()