import pandas as pd
# pyrefly: ignore [missing-import]
import seaborn as sns
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from statsmodels.stats.outliers_influence import variance_inflation_factor
# pyrefly: ignore [missing-import]
from statsmodels.tools.tools import add_constant
import os

def run_multicollinearity_check():
    print("[-] Running Multicollinearity & VIF Analysis...")
    file_path = "../data/processed/master_dataset.csv"

    if not os.path.exists(file_path):
        print("[!] Error: master_dataset.csv not found.")
        return

    df = pd.read_csv(file_path)

    # Fix 1: Drop the '0192' outlier rows BEFORE date parsing
    # These are clearly corrupted entries (year 192 AD)
    df = df[~df['Date'].str.startswith('0192')].copy()

    # Fix 2: Parse mixed date formats — handle DD/MM/YYYY and YYYY-MM-DD
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Fix 3: Drop any rows where date parsing still failed
    df = df[df['Date'].notna()].copy()

    print(f"[✓] {len(df)} rows loaded | Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    # Isolate the exogenous features (independent variables)
    features = ['Inflation_Rate', 'UK_Vacancies_Thousands', 'GDP_Value_mil', 'BoE_Base_Rate']
    df_features = df[features].dropna()

    print(f"[✓] {len(df_features)} rows available after dropping NaNs")

    if df_features.empty:
        print("[!] No data remaining after dropna(). Aborting.")
        return

    # --- PLOT 1: Correlation Matrix Heatmap ---
    plt.figure(figsize=(8, 6))
    correlation_matrix = df_features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # --- CALCULATION: Variance Inflation Factor (VIF) ---
    X = add_constant(df_features)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF_Score"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    print("\n=== Variance Inflation Factor (VIF) Scores ===")
    print("Rule of Thumb: VIF > 5 = moderate concern | VIF > 10 = severe multicollinearity")
    print(vif_data[vif_data["Feature"] != "const"].to_string(index=False))

if __name__ == "__main__":
    run_multicollinearity_check()