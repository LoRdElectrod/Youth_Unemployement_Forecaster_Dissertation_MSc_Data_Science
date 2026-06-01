import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib.dates as mdates
import seaborn as sns
import os

def run_eda():
    print("[-] Loading master dataset for EDA...")
    file_path = "../data/processed/master_dataset.csv"
    
    if not os.path.exists(file_path):
        print("[!] Error: master_dataset.csv not found.")
        return
        
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime for plotting, coercing errors to NaT
    df['Date_parsed'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')

    # --- PLOT 1: Missing Data Heatmap ---
    # This shows us EXACTLY where the data is missing in the matrix
    plt.figure(figsize=(10, 5))
    cols_to_plot = [c for c in df.columns if c != 'Date_parsed']
    sns.heatmap(df[cols_to_plot].isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Visualizing Missing Values (Yellow = Missing)")
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: The "0192" Outlier Effect ---
    # We will plot the raw dates to see how the typo ruins the timeline
    df['Year_num'] = df['Date_parsed'].dt.year.astype('float')

    plt.figure(figsize=(12, 5))
    for region, group in df.groupby('Region'):
        plt.scatter(group['Year_num'], group['Youth_Unemployment_Rate'],
                    label=region, s=30)

    plt.xlabel('Date_parsed')
    plt.ylabel('Youth_Unemployment_Rate')
    plt.title('Youth Unemployment Timeline (Spot the Outlier!)')
    plt.legend(title='Region')
    plt.tight_layout()
    plt.show()

    # --- PLOT 3: The GDP Lag (Zooming in on the 2000s) ---
    # Filtering out the bad date to see the GDP cutoff clearly
    df_clean_dates = df[df['Date_parsed'].dt.year > 1990]
    
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df_clean_dates, x='Date_parsed', y='GDP_Value_mil', hue='Region', marker="o")
    plt.title("Regional GDP Over Time (Notice the drop-off at the end)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda()