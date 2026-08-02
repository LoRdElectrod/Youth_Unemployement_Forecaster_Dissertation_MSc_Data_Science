import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

def run_multicollinearity_check():
    print("[-] Running Multicollinearity & VIF Analysis (Including RTI Data)...")
    file_path = "../data/processed/master_dataset.csv"

    if not os.path.exists(file_path):
        print("[!] Error: master_dataset.csv not found.")
        return

    df = pd.read_csv(file_path)

    # 1. Drop the '0192' outlier rows BEFORE date parsing
    df = df[~df['Date'].str.startswith('0192')].copy()

    # 2. Parse mixed date formats
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # 3. Drop any rows where date parsing failed
    df = df[df['Date'].notna()].copy()

    # 4. Isolate the exogenous features (including the new RTI feature)
    # NOTE: Ensure 'RTI_Payrolled_Employees' exists in your merged CSV
    features = ['Inflation_Rate', 'UK_Vacancies_Thousands', 'GDP_Value_mil', 'BoE_Base_Rate', 'RTI_Payrolled_Employees']
    
    # Check if the new column is present
    if 'RTI_Payrolled_Employees' not in df.columns:
        print("[!] Error: 'RTI_Payrolled_Employees' column missing. Please merge the RTI data first.")
        return

    df_features = df[features].dropna()

    if df_features.empty:
        print("[!] No data remaining after dropna(). Aborting.")
        return

    # --- PLOT 1: Correlation Matrix Heatmap ---
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix (with HMRC RTI)")
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