import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Suppress statsmodels convergence and other warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PLOTS = False

def save_and_show_modeling_plot(filename, doc_fig_path):
    plt.savefig(doc_fig_path, dpi=300)
    if SAVE_PLOTS:
        plot_dir = os.path.join(PROJECT_ROOT, "PLOT_AFTER")
        os.makedirs(plot_dir, exist_ok=True)
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[+] Saved plot to {filepath}")
    else:
        plt.show()

def run_modeling_pipeline():
    print("[-] Initializing Machine Learning & Econometric Modeling Pipeline...")
    file_path = os.path.join(PROJECT_ROOT, "data", "processed", "ml_matrix.csv")
    
    if not os.path.exists(file_path):
        print("[!] Error: ml_matrix.csv not found. Please run feature engineering first.")
        return
        
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Baseline features used as exogenous features in Prophet and SARIMAX
    exog_features = ['GDP_Value_mil', 'Inflation_Rate', 'UK_Vacancies_Thousands', 'BoE_Base_Rate']
    
    # XGBoost rich features (includes lags and cyclical quarterly sine/cosine features)
    xgb_features = [
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
    
    # Demographic mappings for Human Impact translation
    labor_force_map = {
        "London": 600000,
        "North East": 150000
    }
    
    comparison_records = []
    
    for region in regions:
        print(f"\n==========================================")
        print(f"  MODELING REGION: {region}")
        print(f"==========================================")
        
        region_df = df[df['Region'] == region].copy()
        
        # --- TRAIN/TEST SPLIT (Train up to 2023, Test on 2024-2025) ---
        train = region_df[region_df['Date'].dt.year <= 2023].copy()
        test = region_df[region_df['Date'].dt.year >= 2024].copy()
        
        if len(test) == 0:
            print(f"[!] Warning: No test data available for 2024-2025 in region {region}. Skipping.")
            continue
            
        print(f"[*] Train set: {len(train)} quarters | Test set: {len(test)} quarters")
        
        # ----------------------------------------------------
        # 1. FB PROPHET BASELINE MODEL
        # ----------------------------------------------------
        print("\n[>] Fitting Prophet baseline model...")
        train_prophet = train.rename(columns={'Date': 'ds', target: 'y'})
        test_prophet = test.rename(columns={'Date': 'ds', target: 'y'})
        
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        for reg in exog_features:
            m_prophet.add_regressor(reg)
            
        m_prophet.fit(train_prophet)
        forecast_prophet = m_prophet.predict(test_prophet)
        
        y_pred_prophet = forecast_prophet['yhat'].values
        mae_prophet = mean_absolute_error(test_prophet['y'], y_pred_prophet)
        rmse_prophet = np.sqrt(mean_squared_error(test_prophet['y'], y_pred_prophet))
        
        print(f"    * Prophet MAE: {mae_prophet:.2f}% | RMSE: {rmse_prophet:.2f}%")
        
        # ----------------------------------------------------
        # 2. SARIMAX ECONOMETRIC MODEL
        # ----------------------------------------------------
        print("\n[>] Fitting SARIMAX model...")
        
        # Set datetime index for temporal frequency checks
        train_sarimax = train.set_index('Date')
        test_sarimax = test.set_index('Date')
        
        inferred_freq = pd.infer_freq(train_sarimax.index)
        if inferred_freq:
            train_sarimax.index.freq = inferred_freq
            test_sarimax.index.freq = inferred_freq
            
        y_train_sarimax = train_sarimax[target]
        exog_train = train_sarimax[exog_features]
        y_test_sarimax = test_sarimax[target]
        exog_test = test_sarimax[exog_features]
        
        # Check stationarity via ADF
        adf_result = adfuller(y_train_sarimax.dropna())
        p_value = adf_result[1]
        d_term = 0 if p_value < 0.05 else 1
        print(f"    * ADF p-value: {p_value:.4f} -> Dynamically setting d={d_term}")
        
        # Scale exogenous variables
        scaler = StandardScaler()
        exog_train_scaled = scaler.fit_transform(exog_train)
        exog_test_scaled = scaler.transform(exog_test)
        
        exog_train_scaled = pd.DataFrame(exog_train_scaled, columns=exog_features, index=exog_train.index)
        exog_test_scaled = pd.DataFrame(exog_test_scaled, columns=exog_features, index=exog_test.index)
        
        m_sarimax = SARIMAX(
            endog=y_train_sarimax,
            exog=exog_train_scaled,
            order=(1, d_term, 1),
            seasonal_order=(1, 1, 1, 4),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        res_sarimax = m_sarimax.fit(disp=False)
        y_pred_sarimax = res_sarimax.forecast(steps=len(test_sarimax), exog=exog_test_scaled).values
        
        mae_sarimax = mean_absolute_error(y_test_sarimax, y_pred_sarimax)
        rmse_sarimax = np.sqrt(mean_squared_error(y_test_sarimax, y_pred_sarimax))
        print(f"    * SARIMAX MAE: {mae_sarimax:.2f}% | RMSE: {rmse_sarimax:.2f}%")
        
        # ----------------------------------------------------
        # 3. XGBOOST GBDT MODEL
        # ----------------------------------------------------
        print("\n[>] Fitting XGBoost model...")
        X_train, y_train = train[xgb_features], train[target]
        X_test, y_test = test[xgb_features], test[target]
        
        m_xgb = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            objective='reg:squarederror'
        )
        
        m_xgb.fit(X_train, y_train)
        y_pred_xgb = m_xgb.predict(X_test)
        
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        print(f"    * XGBoost MAE: {mae_xgb:.2f}% | RMSE: {rmse_xgb:.2f}%")
        
        # Save evaluation metrics
        comparison_records.append({
            "Region": region,
            "Prophet_MAE": mae_prophet, "Prophet_RMSE": rmse_prophet,
            "SARIMAX_MAE": mae_sarimax, "SARIMAX_RMSE": rmse_sarimax,
            "XGBoost_MAE": mae_xgb, "XGBoost_RMSE": rmse_xgb
        })
        
        # --- VISUALIZATION 1: Forecast Comparison ---
        print("[-] Saving forecast comparisons to docs/figures/...")
        docs_fig_dir = os.path.join(PROJECT_ROOT, "docs", "figures")
        os.makedirs(docs_fig_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(train['Date'], train[target], label='Training Data (Actual)', color='black')
        plt.plot(test['Date'], test[target], label='Test Data (Actual)', color='blue', marker='o')
        plt.plot(test['Date'], y_pred_xgb, label='XGBoost Forecast', color='green', linestyle='--', marker='x')
        plt.plot(test['Date'], y_pred_prophet, label='Prophet Forecast', color='red', linestyle=':', marker='.')
        plt.plot(test['Date'], y_pred_sarimax, label='SARIMAX Forecast', color='orange', linestyle='-.', marker='+')
        
        plt.title(f'Model Forecasting Performance Backtest: {region} (2024-2025)')
        plt.xlabel('Date')
        plt.ylabel('Youth Unemployment Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        doc_path = os.path.join(docs_fig_dir, f"forecast_comparison_{region.replace(' ', '_').lower()}.png")
        save_and_show_modeling_plot(f"forecast_comparison_{region.replace(' ', '_').lower()}.png", doc_path)
        
        # --- VISUALIZATION 2: Feature Importance (XGBoost) ---
        plt.figure(figsize=(10, 6))
        importances = pd.Series(m_xgb.feature_importances_, index=xgb_features).sort_values(ascending=True)
        importances.tail(10).plot(kind='barh', color='teal')
        plt.title(f'Top 10 Feature Importances (XGBoost): {region}')
        plt.xlabel('Relative F-Score split-gain')
        plt.tight_layout()
        doc_path = os.path.join(docs_fig_dir, f"xgboost_importance_{region.replace(' ', '_').lower()}.png")
        save_and_show_modeling_plot(f"xgboost_importance_{region.replace(' ', '_').lower()}.png", doc_path)

    # ----------------------------------------------------
    # 4. POLICYMAKER EVALUATION & SUMMARY BENCHMARKS
    # ----------------------------------------------------
    print("\n==========================================")
    print("  SUMMARY BENCHMARKS & CAUSALITY IMPACTS")
    print("==========================================")
    
    df_metrics = pd.DataFrame(comparison_records)
    print("\nStatistical Error Comparisons Matrix:")
    print(df_metrics.to_string(index=False))
    
    print("\n=== Translating Statistical Error into Absolute Human Impact Headcounts ===")
    print("The headcount margin of error is calculated from XGBoost RMSE relative to active youth workforce sizes.")
    
    human_records = []
    for record in comparison_records:
        region = record["Region"]
        labor_force = labor_force_map.get(region, 0)
        rmse_val = record["XGBoost_RMSE"]
        
        # Headcount margin: (RMSE / 100) * Labor Force
        headcount_err = int((rmse_val / 100) * labor_force)
        human_records.append({
            "Region": region,
            "Active Youth Workforce": labor_force,
            "XGBoost RMSE (%)": f"{rmse_val:.2f}%",
            "Absolute Uncertainty (+/- Headcount)": f"{headcount_err:,} young people"
        })
        
        print(f"[*] {region}:")
        print(f"    - Estimated Active Youth Labor Force: {labor_force:,} people")
        print(f"    - XGBoost Backtest RMSE Error Bounds: {rmse_val:.2f}%")
        print(f"    - Absolute Policymaker Error Margin: +/- {headcount_err:,} youth workers")
        
    df_human = pd.DataFrame(human_records)
    
    # Save a CSV benchmark record for transparency
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(processed_dir, "model_benchmarks.csv"), index=False)
    print("\n[+] Model benchmarking metrics saved to data/processed/model_benchmarks.csv")

if __name__ == "__main__":
    run_modeling_pipeline()
