import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_human_impact():
    print("[-] Translating XGBoost RMSE into Absolute Human Impact...")
    
    # Using the exact RMSE values from your XGBoost backtest
    results = {
        'Region': ['London', 'North East'],
        'XGBoost_RMSE_Pct': [3.06, 4.34],
        # Estimated active labor force (16-24) from ONS demographic data
        'Active_Youth_Labor_Force': [600000, 150000] 
    }
    
    df_eval = pd.DataFrame(results)
    
    # The Math: (RMSE / 100) * Labor Force = Actual Headcount Margin of Error
    df_eval['Human_Error_Headcount'] = (df_eval['XGBoost_RMSE_Pct'] / 100) * df_eval['Active_Youth_Labor_Force']
    df_eval['Human_Error_Headcount'] = df_eval['Human_Error_Headcount'].astype(int)
    
    print("\n=== Policymaker Evaluation Metrics ===")
    for index, row in df_eval.iterrows():
        print(f"[*] {row['Region']}:")
        print(f"    - Statistical Error (RMSE): {row['XGBoost_RMSE_Pct']}%")
        print(f"    - Human Impact Error: +/- {row['Human_Error_Headcount']:,} young people")
        
    # --- VISUALIZATION: The Policymaker Chart ---
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_eval, x='Region', y='Human_Error_Headcount', palette=['#1f77b4', '#ff7f0e'])
    
    plt.title('Model Uncertainty: Absolute Human Impact (Ages 16-24)', fontweight='bold', pad=15)
    plt.ylabel('Margin of Error (Number of People)', fontweight='bold')
    plt.xlabel('Region', fontweight='bold')
    
    # Add the exact numbers on top of the bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontweight='bold', color='black', xytext=(0, 5), 
                    textcoords='offset points')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calculate_human_impact()