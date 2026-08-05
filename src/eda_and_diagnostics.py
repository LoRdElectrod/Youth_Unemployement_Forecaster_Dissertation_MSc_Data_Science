import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
from statsmodels.tsa.stattools import ccf, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PLOTS = False

def show_or_save_plot(filename):
    if SAVE_PLOTS:
        plot_dir = os.path.join(PROJECT_ROOT, "PLOT_AFTER")
        os.makedirs(plot_dir, exist_ok=True)
        filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[+] Saved plot to {filepath}")
    else:
        plt.show()

def run_missing_and_outliers_eda():
    print("\n--- [PART 1] Missing Data & Outlier Analysis ---")
    file_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset.csv")
    if not os.path.exists(file_path):
        print("[!] Error: master_dataset.csv not found. Please run data ingestion first.")
        return
        
    df = pd.read_csv(file_path)
    df['Date_parsed'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')

    # Plot 1: Missing Data Heatmap
    print("[-] Displaying Missing Data Heatmap...")
    plt.figure(figsize=(10, 5))
    cols_to_plot = [c for c in df.columns if c != 'Date_parsed']
    sns.heatmap(df[cols_to_plot].isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Visualizing Missing Values (Yellow = Missing)")
    plt.tight_layout()
    show_or_save_plot("missing_data_heatmap.png")

    # Plot 2: Outlier Effect Timeline
    print("[-] Displaying Outlier Timeline...")
    df['Year_num'] = df['Date_parsed'].dt.year.astype('float')
    plt.figure(figsize=(12, 5))
    for region, group in df.groupby('Region'):
        plt.scatter(group['Year_num'], group['Youth_Unemployment_Rate'], label=region, s=30)
    plt.xlabel('Parsed Date')
    plt.ylabel('Youth Unemployment Rate (%)')
    plt.title('Youth Unemployment Timeline (Highlighting Timeline Typos/Outliers)')
    plt.legend(title='Region')
    plt.tight_layout()
    show_or_save_plot("outlier_timeline.png")

    # Plot 3: Spliced GDP Trend (Filtered for display)
    print("[-] Displaying GDP Trends...")
    df_clean_dates = df[df['Date_parsed'].dt.year > 1990]
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df_clean_dates, x='Date_parsed', y='GDP_Value_mil', hue='Region', marker="o")
    plt.title("Regional GDP Over Time (Pre-Splicing Gap Visual)")
    plt.tight_layout()
    show_or_save_plot("gdp_trends.png")


def run_structural_breaks_analysis():
    print("\n--- [PART 2] Structural Breaks Analysis ---")
    file_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset.csv")
    if not os.path.exists(file_path):
        print("[!] Error: master_dataset.csv not found.")
        return
        
    df = pd.read_csv(file_path)
    df = df[~df['Date'].astype(str).str.startswith('0192')].copy()
    df['Date_parsed'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df['Year_num'] = df['Date_parsed'].dt.year.astype('float')
    df_clean = df[df['Year_num'] > 1990].copy()

    print("[-] Visualizing Structural Breaks (Financial Crisis, COVID-19, Austerity)...")
    fig, ax = plt.subplots(figsize=(16, 8))

    # Secondary Shock band behind main lines
    ax.axvspan(2010.0, 2015.0, color='orange', alpha=0.10, zorder=0)

    # Main lines
    sns.lineplot(data=df_clean, x='Year_num', y='Youth_Unemployment_Rate', hue='Region', linewidth=2.5, ax=ax)

    # Structural Break 1: 2008 Financial Crash
    ax.axvspan(2008.5, 2009.8, color='red', alpha=0.15, zorder=1)
    ax.text(2009.15, 25, '2008\nCrash', color='darkred', weight='bold', zorder=2, ha='center', fontsize=9)

    # Structural Break 2: COVID-19
    ax.axvspan(2020.0, 2021.8, color='purple', alpha=0.15, zorder=1)
    ax.text(2020.9, 25, 'COVID-19\nShock', color='purple', weight='bold', zorder=2, ha='center', fontsize=9)

    # Double-headed bracket arrow
    ax.annotate('', xy=(2010.0, 27.8), xytext=(2015.0, 27.8), arrowprops=dict(arrowstyle='<->', color='darkorange', lw=1.8))
    ax.text(2012.5, 28.2, 'Secondary Shock Period (2010–2015)', color='darkorange', weight='bold', zorder=2, ha='center', fontsize=8.5)

    # Annotations
    ax.annotate(
        '① 2010 Austerity Cuts\nPublic sector freeze → N.East\nheavily impacted vs London',
        xy=(2010.5, 21.4), xytext=(2007.2, 10.5),
        fontsize=7.8, color='darkred', ha='center', zorder=3,
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.3, connectionstyle='arc3,rad=0.2'),
        bbox=dict(boxstyle='round,pad=0.4', fc='#fff3f3', ec='darkred', alpha=0.9)
    )

    ax.annotate(
        '② Eurozone Debt Crisis (2011–12)\nGreece/Spain panic froze UK\nhiring budgets → graduate lockout',
        xy=(2012.0, 25.5), xytext=(2011.5, 10.5),
        fontsize=7.8, color='navy', zorder=3, ha='center',
        arrowprops=dict(arrowstyle='->', color='navy', lw=1.3, connectionstyle='arc3,rad=-0.25'),
        bbox=dict(boxstyle='round,pad=0.4', fc='#f0f0ff', ec='navy', alpha=0.9)
    )

    ax.annotate(
        '③ Graduate Bottleneck (Lag Effect)\n3 yrs of school leavers entered\nmarket simultaneously 2011–13',
        xy=(2013.0, 26.4), xytext=(2015.8, 10.5),
        fontsize=7.8, color='darkgreen', zorder=3, ha='center',
        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.3, connectionstyle='arc3,rad=-0.2'),
        bbox=dict(boxstyle='round,pad=0.4', fc='#f0fff0', ec='darkgreen', alpha=0.9)
    )

    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Youth Unemployment Rate (%)', fontweight='bold')
    ax.set_title(
        'Macroeconomic Anomalies & Structural Breaks in UK Youth Unemployment (2001–2025)\n'
        'Annotated with Secondary Shock Drivers: Austerity, Eurozone Crisis & Graduate Bottleneck',
        fontsize=13, fontweight='bold', pad=15
    )

    # Clean manual legend
    london_patch = mpatches.Patch(color='steelblue', label='London')
    northeast_patch = mpatches.Patch(color='orange', label='North East')
    red_patch = mpatches.Patch(color='red', alpha=0.3, label='Structural Break: 2008 Financial Crash')
    purple_patch = mpatches.Patch(color='purple', alpha=0.3, label='Structural Break: COVID-19 Pandemic')
    orange_patch = mpatches.Patch(color='orange', alpha=0.3, label='Secondary Shock: Austerity & Eurozone Crisis (2010–2015)')

    ax.legend(handles=[london_patch, northeast_patch, red_patch, purple_patch, orange_patch], title='Legend', loc='upper left', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(9, 30)
    plt.tight_layout()
    show_or_save_plot("structural_breaks_analysis.png")


def run_multicollinearity_and_vif():
    print("\n--- [PART 3] Multicollinearity & VIF Analysis ---")
    file_path = os.path.join(PROJECT_ROOT, "data", "processed", "master_dataset.csv")
    if not os.path.exists(file_path):
        print("[!] Error: master_dataset.csv not found.")
        return

    df = pd.read_csv(file_path)
    df = df[~df['Date'].astype(str).str.startswith('0192')].copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df[df['Date'].notna()].copy()

    features = ['Inflation_Rate', 'UK_Vacancies_Thousands', 'GDP_Value_mil', 'BoE_Base_Rate', 'RTI_Payrolled_Employees']
    
    if 'RTI_Payrolled_Employees' not in df.columns:
        print("[!] Error: 'RTI_Payrolled_Employees' column missing. Please ingestion correct data.")
        return

    df_features = df[features].dropna()

    if df_features.empty:
        print("[!] No data remaining after dropping nulls for VIF calculation.")
        return

    # Correlation Matrix Heatmap
    print("[-] Visualizing correlation matrix...")
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix (including HMRC RTI)")
    plt.tight_layout()
    show_or_save_plot("correlation_matrix.png")

    # VIF check
    X = add_constant(df_features)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF_Score"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    print("\n=== Variance Inflation Factor (VIF) Scores ===")
    print("Rule of Thumb: VIF > 5 = moderate concern | VIF > 10 = severe multicollinearity")
    print(vif_data[vif_data["Feature"] != "const"].to_string(index=False))


def run_causality_diagnostics():
    print("\n--- [PART 4] Causality Diagnostics (CCF & Granger) ---")
    file_path = os.path.join(PROJECT_ROOT, "data", "processed", "ml_matrix.csv")
    if not os.path.exists(file_path):
        print("[!] Error: ml_matrix.csv not found. Please run feature engineering first.")
        return
        
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Isolate London for Granger test consistency
    region_df = df[df['Region'] == 'London'].copy()
    region_df.set_index('Date', inplace=True)
    region_df.sort_index(inplace=True)

    us_metric = 'US_NFP' 
    uk_metric = 'Youth_Unemployment_Rate' 

    if us_metric not in region_df.columns:
        print(f"[!] {us_metric} not found in matrix. Please execute feature engineering with US variables.")
        return

    test_data = region_df[[uk_metric, us_metric]].dropna()

    # 1. Cross-Correlation Function (CCF)
    print(f"\n[>] 1. Cross-Correlation Function (CCF) - London")
    print(f"Testing how historical shifts in {us_metric} correlate with future {uk_metric} (up to 8 quarters lag).")
    ccf_values = ccf(test_data[uk_metric], test_data[us_metric])[:9] 
    
    for lag, val in enumerate(ccf_values):
        print(f"    * Lag {lag} Quarters: Correlation = {val:.4f}")

    # Plot CCF
    plt.figure(figsize=(10, 5))
    plt.stem(range(len(ccf_values)), ccf_values, basefmt=" ")
    plt.axhline(0, color='black', linewidth=1)
    plt.axhline(1.96 / np.sqrt(len(test_data)), color='red', linestyle='--', label='95% Confidence Interval')
    plt.axhline(-1.96 / np.sqrt(len(test_data)), color='red', linestyle='--')
    plt.title(f'Cross-Correlation: {us_metric} vs {uk_metric} (London)')
    plt.xlabel('Lag (Quarters)')
    plt.ylabel('Correlation Coefficient')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    show_or_save_plot("cross_correlation_us_nfp_vs_youth_unemployment.png")

    # 2. Granger Causality Test
    print(f"\n[>] 2. Granger Causality Test - London")
    print(f"Hypothesis: Past values of {us_metric} help predict {uk_metric}.")
    print("If p-value < 0.05, we reject the null hypothesis (i.e. US data Granger-causes UK data).")
    
    max_lags = 4
    try:
        granger_results = grangercausalitytests(test_data[[uk_metric, us_metric]], maxlag=max_lags, verbose=False)
        for lag in range(1, max_lags + 1):
            p_val = granger_results[lag][0]['ssr_ftest'][1]
            significance = "SIGNIFICANT (Keep Feature)" if p_val < 0.05 else "Not Significant (Discard)"
            print(f"    * Lag {lag} Quarters -> p-value: {p_val:.4f} | {significance}")
    except Exception as e:
        print(f"[!] Granger test failed: {e}")


def main():
    print("====================================================")
    print("  EXPLORATORY DATA ANALYSIS & CAUSALITY DIAGNOSTICS")
    print("====================================================")
    run_missing_and_outliers_eda()
    run_structural_breaks_analysis()
    run_multicollinearity_and_vif()
    run_causality_diagnostics()
    print("\n[+] EDA & Diagnostics pipeline run completed successfully.")

if __name__ == "__main__":
    main()
