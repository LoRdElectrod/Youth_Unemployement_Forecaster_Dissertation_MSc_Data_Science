import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    # Like Prophet, SARIMAX handles its own auto-regression, so we do not use the XGBoost lag features here.
    exog_features = ['GDP_Value_mil', 'Inflation_Rate', 'UK_Vacancies_Thousands', 'BoE_Base_Rate']
    target = 'Youth_Unemployment_Rate'
    
    regions = df['Region'].unique()
    
    for region in regions:
        print(f"\n[>] Training SARIMAX Model for: {region}")
        
        # [FIX 1]: Reset the index immediately after filtering to prevent statsmodels shape/alignment errors
        region_df = df[df['Region'] == region].copy().reset_index(drop=True)
        
        # --- TRAIN/TEST SPLIT ---
        train = region_df[region_df['Date'].dt.year <= 2023].copy()
        test = region_df[region_df['Date'].dt.year >= 2024].copy()
        
        y_train = train[target]
        exog_train = train[exog_features]
        
        y_test = test[target]
        exog_test = test[exog_features]
        
        # --- MODEL CONFIGURATION ---
        # order = (p, d, q) -> Auto-Regressive, Differencing, Moving Average
        # seasonal_order = (P, D, Q, s) -> s=4 for quarterly data
        # Note: These parameters are a standard baseline. In a real thesis, you might mention using 'auto_arima' to find optimal p,d,q.
        model = SARIMAX(
            endog=y_train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 4),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit the model
        print("    * Fitting model (this may take a moment)...")
        results = model.fit(disp=False)
        
        # [FIX 2]: Generate predictions using .forecast() instead of .predict()
        forecast = results.forecast(steps=len(test), exog=exog_test)
        
        # --- EVALUATION ---
        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))
        
        print(f"    * SARIMAX MAE: {mae:.2f}%")
        print(f"    * SARIMAX RMSE: {rmse:.2f}%")
        
        # --- VISUALIZATION ---
        plt.figure(figsize=(10, 5))
        plt.plot(train['Date'], train[target], label='Training Data', color='black')
        plt.plot(test['Date'], test[target], label='Actual Data (2024-2025)', color='blue', marker='o')
        plt.plot(test['Date'], forecast, label='SARIMAX Forecast', color='orange', linestyle='--', marker='x')
        
        plt.title(f'SARIMAX Backtest: {region} (2024-2025)')
        plt.xlabel('Date')
        plt.ylabel('Youth Unemployment Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()

if __name__ == "__main__":
    run_sarimax_model()