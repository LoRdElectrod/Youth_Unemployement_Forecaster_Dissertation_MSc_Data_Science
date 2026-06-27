import pandas as pd
import numpy as np
import os

def engineer_features():
    print("[-] Initiating Feature Engineering & GDP Splicing...")
    input_path = "../data/processed/master_dataset.csv"
    output_path = "../data/processed/ml_matrix.csv"
    
    if not os.path.exists(input_path):
        print("[!] Error: master_dataset.csv not found.")
        return
        
    df = pd.read_csv(input_path)
    
    # 1. Clean the 0192 outlier and parse dates (using your proven logic)
    df = df[~df['Date'].str.startswith('0192')].copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df[df['Date'].notna()].copy()
    df = df.sort_values(by=['Region', 'Date']).reset_index(drop=True)

    # 2. Institutional Growth Rate Splicing (GDP)
    # Annual growth of 0.9% = (1.009)^(1/4) - 1 for quarterly compounding
    q_growth_rate = (1.009 ** 0.25) - 1 
    
    def splice_gdp_series(series):
        s = series.copy()
        last_valid_idx = s.last_valid_index()
        if last_valid_idx is not None:
            last_val = s.loc[last_valid_idx]
            missing_mask = s.isna() & (s.index > last_valid_idx)
            
            # Apply compounding growth to the missing quarters
            periods_missing = np.arange(1, missing_mask.sum() + 1)
            s.loc[missing_mask] = last_val * ((1 + q_growth_rate) ** periods_missing)
        return s

    # Apply the splicing per region just to the GDP column
    df['GDP_Value_mil'] = df.groupby('Region')['GDP_Value_mil'].transform(splice_gdp_series)
    print(f"[✓] Successfully spliced trailing GDP using 0.9% OBR/IMF annual forecast.")

    # 3. Engineer Lag Features (The "Delay Effect")
    print("[-] Generating Lag Features for XGBoost...")
    lags = {
        'Youth_Unemployment_Rate': [1, 4], # Last quarter, and same quarter last year
        'GDP_Value_mil': [1, 4],
        'Inflation_Rate': [1, 4],
        'UK_Vacancies_Thousands': [1, 4],
        'BoE_Base_Rate': [1, 2, 4] # Interest rates have longer delay impacts
    }
    
    for col, lag_list in lags.items():
        for lag in lag_list:
            df[f'{col}_Lag_{lag}'] = df.groupby('Region')[col].shift(lag)
            
    # 4. Cyclical Encoding for Seasonality
    df['Quarter'] = df['Date'].dt.quarter
    df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
    df = df.drop(columns=['Quarter'])

    # 5. Final Cleanup
    # Shifting creates NaNs at the very beginning of the dataset
    # We will drop these initial rows so XGBoost gets a perfectly clean matrix
    ml_matrix = df.dropna().copy()
    
    os.makedirs("../data/processed", exist_ok=True)
    ml_matrix.to_csv(output_path, index=False)
    
    print(f"[+] Success! Machine Learning Matrix saved to: {output_path}")
    print(f"[+] Final Matrix Shape: {ml_matrix.shape[0]} rows, {ml_matrix.shape[1]} columns")
    
if __name__ == "__main__":
    engineer_features()