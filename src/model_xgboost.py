import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

def run_xgboost_model():
    print("[-] Initializing XGBoost Modeling...")
    file_path = "../data/processed/ml_matrix.csv"
    
    if not os.path.exists(file_path):
        print("[!] Error: ml_matrix.csv not found.")
        return
        
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define our rich feature matrix
    features = [
        'GDP_Value_mil', 'Inflation_Rate', 'UK_Vacancies_Thousands', 'BoE_Base_Rate',
        'Youth_Unemployment_Rate_Lag_1', 'Youth_Unemployment_Rate_Lag_4',
        'GDP_Value_mil_Lag_1', 'GDP_Value_mil_Lag_4',
        'Inflation_Rate_Lag_1', 'Inflation_Rate_Lag_4',
        'UK_Vacancies_Thousands_Lag_1', 'UK_Vacancies_Thousands_Lag_4',
        'BoE_Base_Rate_Lag_1', 'BoE_Base_Rate_Lag_2', 'BoE_Base_Rate_Lag_4',
        'Quarter_Sin', 'Quarter_Cos'
    ]
    
    target = 'Youth_Unemployment_Rate'
    regions = df['Region'].unique()
    
    for region in regions:
        print(f"\n[>] Training XGBoost Model for: {region}")
        region_df = df[df['Region'] == region].copy()
        
        # --- TRAIN/TEST SPLIT ---
        # Strictly identical to the Prophet backtest for a fair comparison
        train = region_df[region_df['Date'].dt.year <= 2023].copy()
        test = region_df[region_df['Date'].dt.year >= 2024].copy()
        
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]
        
        # Initialize the Gradient Boosted Tree
        model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            objective='reg:squarederror'
        )
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Generate Predictions
        y_pred = model.predict(X_test)
        
        # --- EVALUATION ---
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"    * XGBoost MAE: {mae:.2f}%")
        print(f"    * XGBoost RMSE: {rmse:.2f}%")
        
        # --- VISUALIZATION 1: Forecast vs Actual ---
        plt.figure(figsize=(10, 5))
        plt.plot(train['Date'], train[target], label='Training Data', color='black')
        plt.plot(test['Date'], test[target], label='Actual Data (2024-2025)', color='blue', marker='o')
        plt.plot(test['Date'], y_pred, label='XGBoost Forecast', color='green', linestyle='--', marker='x')
        
        plt.title(f'XGBoost Backtest: {region} (2024-2025)')
        plt.xlabel('Date')
        plt.ylabel('Youth Unemployment Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs("../docs/figures", exist_ok=True)
        plt.savefig(f"../docs/figures/xgboost_backtest_{region.replace(' ', '_').lower()}.png", dpi=300)
        plt.show()
        
        # --- VISUALIZATION 2: Feature Importance ---
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        
        # Plot only the top 10 most influential features to keep the chart clean
        importances.tail(10).plot(kind='barh', color='teal')
        plt.title(f'Top 10 Macroeconomic Drivers (XGBoost): {region}')
        plt.xlabel('Relative Importance (F-Score)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_xgboost_model()