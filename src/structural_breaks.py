import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import matplotlib.patches as mpatches
import seaborn as sns
import os

def visualize_structural_breaks():
    print("[-] Generating Structural Break Visualizations...")
    file_path = "../data/processed/master_dataset.csv"
    
    if not os.path.exists(file_path):
        print("[!] Error: master_dataset.csv not found.")
        return
        
    df = pd.read_csv(file_path)
    
    df['Date_parsed'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df['Year_num'] = df['Date_parsed'].dt.year.astype('float')
    df_clean = df[df['Year_num'] > 1990].copy()

    fig, ax = plt.subplots(figsize=(16, 8))

    # --- NEW: Secondary Shock band FIRST so it sits behind everything ---
    ax.axvspan(2010.0, 2015.0, color='orange', alpha=0.10, zorder=0)

    # --- Main lines ---
    sns.lineplot(data=df_clean, x='Year_num', y='Youth_Unemployment_Rate',
                 hue='Region', linewidth=2.5, ax=ax)

    # --- Structural Break 1: 2008 Financial Crash ---
    ax.axvspan(2008.5, 2009.8, color='red', alpha=0.15, zorder=1)
    ax.text(2009.15, 25, '2008\nCrash', color='darkred', weight='bold', ha='center', fontsize=9)

    # --- Structural Break 2: COVID-19 ---
    ax.axvspan(2020.0, 2021.8, color='purple', alpha=0.15, zorder=1)
    ax.text(2020.9, 25, 'COVID-19\nShock', color='purple', weight='bold', ha='center', fontsize=9)

    # --- Double-headed bracket arrow across the secondary shock peak ---
    ax.annotate('', xy=(2010.0, 27.8), xytext=(2015.0, 27.8),
                arrowprops=dict(arrowstyle='<->', color='darkorange', lw=1.8))
    ax.text(2012.5, 28.2, 'Secondary Shock Period (2010–2015)',
            color='darkorange', weight='bold', ha='center', fontsize=8.5)

    # --- Annotation 1: Austerity Cuts ---
    ax.annotate(
        '① 2010 Austerity Cuts\nPublic sector freeze → N.East\nheavily impacted vs London',
        xy=(2010.5, 21.4),
        xytext=(2007.2, 10.5),
        fontsize=7.8, color='darkred', ha='center',
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.3,
                        connectionstyle='arc3,rad=0.2'),
        bbox=dict(boxstyle='round,pad=0.4', fc='#fff3f3', ec='darkred', alpha=0.9)
    )

    # --- Annotation 2: Eurozone Debt Crisis ---
    ax.annotate(
        '② Eurozone Debt Crisis (2011–12)\nGreece/Spain panic froze UK\nhiring budgets → graduate lockout',
        xy=(2012.0, 25.5),
        xytext=(2011.5, 10.5),   # ← was (2013.8, 10.5)
        fontsize=7.8, color='navy', ha='center',
        arrowprops=dict(arrowstyle='->', color='navy', lw=1.3,
                        connectionstyle='arc3,rad=-0.25'),
        bbox=dict(boxstyle='round,pad=0.4', fc='#f0f0ff', ec='navy', alpha=0.9)
    )

    # --- Annotation 3: Graduate Bottleneck (Slightly adjusted for space) ---
    ax.annotate(
        '③ Graduate Bottleneck (Lag Effect)\n3 yrs of school leavers entered\nmarket simultaneously 2011–13',
        xy=(2013.0, 26.4),
        xytext=(2015.8, 10.5),   # Slightly adjusted
        fontsize=7.8, color='darkgreen', ha='center',
        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.3,
                        connectionstyle='arc3,rad=-0.2'),   # ← reduced rad slightly
        bbox=dict(boxstyle='round,pad=0.4', fc='#f0fff0', ec='darkgreen', alpha=0.9)
    )

    # --- Formatting ---
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Youth Unemployment Rate (%)', fontweight='bold')
    ax.set_title(
        'Macroeconomic Anomalies & Structural Breaks in UK Youth Unemployment (2001–2025)\n'
        'Annotated with Secondary Shock Drivers: Austerity, Eurozone Crisis & Graduate Bottleneck',
        fontsize=13, fontweight='bold', pad=15
    )

    # --- Clean manual legend (no duplicates) ---
    london_patch = mpatches.Patch(color='steelblue', label='London')
    northeast_patch = mpatches.Patch(color='orange', label='North East')
    red_patch = mpatches.Patch(color='red', alpha=0.3, label='Structural Break: 2008 Financial Crash')
    purple_patch = mpatches.Patch(color='purple', alpha=0.3, label='Structural Break: COVID-19 Pandemic')
    orange_patch = mpatches.Patch(color='orange', alpha=0.3,
                                  label='Secondary Shock: Austerity & Eurozone Crisis (2010–2015)')

    ax.legend(handles=[london_patch, northeast_patch, red_patch, purple_patch, orange_patch],
              title='Legend', loc='upper left', fontsize=8)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(9, 30)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_structural_breaks()