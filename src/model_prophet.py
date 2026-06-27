import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

def run_prophet_baseline():
    print("[-] Initializing Prophet Baseline Modeling...")
    file_path = "../data/processed/ml_matrix.csv"
    
    if not os.path.exists(file_path):
        print("[!] Error: ml_matrix.csv not found.")
        return
        
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prophet strictly requires columns to be named 'ds' (datetime) and 'y' (target)
    df = df.rename(columns={'Date': 'ds', 'Youth_Unemployment_Rate': 'y'})
    
    # Define features to use as external regressors
    # Note: We do NOT use the XGBoost Lag features here. Prophet handles auto-regression internally.
    regressors = ['GDP_Value_mil', 'Inflation_Rate', 'UK_Vacancies_Thousands', 'BoE_Base_Rate']
    
    # We will loop through both regions to build independent models
    regions = df['Region'].unique()
    
    for region in regions:
        print(f"\n[>] Training Prophet Model for: {region}")
        region_df = df[df['Region'] == region].copy()
        
        # --- TRAIN/TEST SPLIT ---
        # Train on 2001 to 2023. Test on 2024 to 2025.
        train = region_df[region_df['ds'].dt.year <= 2023].copy()
        test = region_df[region_df['ds'].dt.year >= 2024].copy()
        
        # Initialize Prophet (allowing it to find yearly seasonality)
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        
        # Add our macroeconomic features
        for reg in regressors:
            m.add_regressor(reg)
            
        # Fit the model
        m.fit(train)
        
        # Predict on the test set
        # We pass the test dataframe because it contains the known 2024/2025 regressor values
        forecast = m.predict(test)
        
        # --- EVALUATION ---
        y_true = test['y'].values
        y_pred = forecast['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"    * Mean Absolute Error (MAE): {mae:.2f}%")
        print(f"    * Root Mean Squared Error (RMSE): {rmse:.2f}%")
        
        # --- VISUALIZATION ---
        plt.figure(figsize=(10, 5))
        plt.plot(train['ds'], train['y'], label='Training Data (Actual)', color='black')
        plt.plot(test['ds'], test['y'], label='Test Data (Actual)', color='blue', marker='o')
        plt.plot(test['ds'], y_pred, label='Prophet Forecast', color='red', linestyle='--', marker='x')
        
        plt.title(f'Prophet Backtest: {region} (2024-2025)')
        plt.xlabel('Date')
        plt.ylabel('Youth Unemployment Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()

if __name__ == "__main__":
    run_prophet_baseline()