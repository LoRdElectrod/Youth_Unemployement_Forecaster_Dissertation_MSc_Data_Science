# UK Youth Unemployment Forecasting & Business Intelligence Dashboard
### *A Regional Comparative Study (London vs. North East) Using Econometrics, Additive Seasonality, and Gradient-Boosted Trees (Ages 16-24)*

---

## 📌 Executive Project Overview
This repository contains the complete data engineering pipeline, predictive modeling suite, and business intelligence (BI) dashboard developed for a MSc Data Science dissertation. The study addresses a critical policy challenge: **forecasting UK youth unemployment (ages 16–24) and mapping regional labor vulnerabilities.** 

By comparing two economically distinct regions—**London** (a services-dominated economy) and the **North East** (an economy historically reliant on public sector employment)—this project analyzes how macroeconomic drivers propagate differently across geographical contexts.

### Research Paradigm
The project contrasts three distinct modeling paradigms:
1. **Traditional Econometrics:** SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors)
2. **Additive Seasonal Modeling:** Facebook Prophet (optimized for structural time series with covariates)
3. **Machine Learning Gradient Boosting:** XGBoost (Gradient-Boosted Decision Trees with engineered lag profiles)

---

## 🛠️ Macroeconomic Variables & Data Dictionary
To capture labor market dynamics, the target variable is mapped against four primary exogenous drivers:

| Macroeconomic Vector | Dataset Feature Name | Source | Description |
| :--- | :--- | :--- | :--- |
| **The Target** | `Youth_Unemployment_Rate` | Office for National Statistics (ONS) | Rolling 3-month average of youth unemployment (16-24) mapped to quarter-ends. |
| **The Output** | `GDP_Value_mil` | ONS Regional Accounts | Regional Gross Domestic Product (real terms, millions of GBP). |
| **The Squeeze** | `Inflation_Rate` | ONS Consumer Price Indices | Mean quarterly UK CPI Annual Rate (%) reflecting consumer pressure. |
| **The Demand** | `UK_Vacancies_Thousands` | ONS Vacancy Survey | National labor demand indicator (total vacancies in thousands). |
| **The Budget** | `BoE_Base_Rate` | Bank of England (BoE) | Official cost of borrowing dictating corporate hiring budgets. |

---

## 🔄 Data Pipeline Architecture

```mermaid
graph TD
    A[Raw Data: ONS & BoE CSVs/XLSX] -->|ingestion.py| B[Merge and Temporal Standardization]
    B -->|master_dataset.csv| C[eda.py & vif_check.py]
    C -->|Identify Outliers & Multicollinearity| D[features.py]
    D -->|Institutional Growth Rate Splicing & Lag Generation| E[ml_matrix.csv]
    E -->|Model Training & Backtesting| F[model_sarimax.py / model_prophet.py / model_xgboost.py]
    F -->|rmse_map & labor_force_map| G[human_impact_eval.py]
    G -->|Interactive Execution & Ollama RAG| H[Streamlit BI Dashboard: app.py]
```

### 1. Ingestion & Temporal Standardization ([ingestion.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/ingestion.py))
- **Temporal Alignment:** Combines datasets reported at mismatched frequencies (annual GDP, monthly CPI, rolling 3-month average unemployment, and irregular BoE rate adjustments) by standardizing to a unified quarterly timeline (representing dates on `-03-31`, `-06-30`, `-09-30`, and `-12-31`).
- **ASOF Joining:** Integrates Bank of England rate changes dynamically, fetching the rate active at the exact close of each quarter using Polars' `join_asof` backward strategy.

### 2. Data Cleaning & Splicing ([features.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/features.py))
- **Anomalous Outlier Removal:** Fixes a critical historical formatting anomaly where a typo representing the year "0192" compressed the timeline.
- **Institutional Growth Rate Splicing (IGRS):** The ONS publishes regional GDP with a multi-year lag. To prevent artificial bias or unrealistic trend-line extrapolation, the pipeline compounds the last recorded GDP value across missing trailing quarters (2024–2025) at a quarterly rate of **0.22%**, representing the official annual economic growth projection of **0.9%** forecasted by the Office for Budget Responsibility (OBR) and IMF:
$$\text{Quarterly Growth Rate} = (1 + 0.009)^{0.25} - 1 \approx 0.00224$$
- **Lag Profiles:** Generates 1-quarter and 4-quarter lags for features to capture delayed macroeconomic effects. For the BoE Base Rate, a 2-quarter lag is also generated to account for interest rate monetary transmission delays.
- **Cyclical Seasonality:** Encodes seasonality using sine and cosine transformations of the quarterly timeline to preserve temporal continuity.

---

## 📊 Exploratory Data Analysis & Multicollinearity
The Exploratory Data Analysis ([eda.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/eda.py)) and Multicollinearity Check ([vif_check.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/vif_check.py)) verified the structural integrity of the feature matrix before modeling.

### Missing Data Visualization
The missing data heatmap confirms that our primary exogenous parameters (CPI, vacancies, and interest rates) are fully populated across the timeline, leaving only the trailing GDP years (pre-splicing) with missing values.

![Missing Value Heatmap](final_plots/missing_val.png)

### Outlier Detection
Plotting the raw timeline highlights the extreme "0192" date typo outlier that compresses the true time series:

![Outlier Scatter Plot](final_plots/outlier.png)

### GDP Trailing Lag
The lineplot shows the abrupt end-points in the raw GDP records, demonstrating the necessity of the Institutional Growth Rate Splicing approach:

![GDP Over Time](final_plots/Regional_GDP_overtime.png)

### Correlation Matrix & VIF Scores
A Variance Inflation Factor (VIF) analysis was run on the exogenous variables to rule out multicollinearity:

![Feature Correlation Matrix](final_plots/Correlation_Matrix.png)

**VIF Scores:**
- **Inflation Rate:** 1.43
- **UK Vacancies (Thousands):** 1.42
- **GDP Value (m):** 1.02
- **BoE Base Rate:** 1.03

*Interpretation:* All features return VIF scores well below the conservative threshold of 5.0, confirming that the explanatory variables are structurally independent and can be safely modeled simultaneously.

---

## 📈 Macroeconomic Anomalies & Structural Breaks ([structural_breaks.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/structural_breaks.py))
A central contribution of the research is analyzing the structural break impact of major macroeconomic shocks on youth employment patterns.

![Structural Breaks Timeline](final_plots/Structural_breaks.png)

### The 2010–2015 "Secondary Shock" Analysis
Beyond the immediate spikes of the **2008 Financial Crash** and the **COVID-19 Pandemic**, the analysis identifies a severe, prolonged **Secondary Shock Period (2010–2015)** driven by three main factors:
1. **2010 Austerity Measures:** Massive public sector spending cuts disproportionately affected the North East (which relies heavily on public administration and service jobs) while London’s private tech/finance sectors rebounded rapidly.
2. **Eurozone Debt Crisis (2011–2012):** Market panic forced UK firms to lock down capital budgets, causing a graduate and school-leaver hiring freeze.
3. **The Graduate Bottleneck:** Delayed labor entry. When the crash hit in 2008, many young people remained in education to escape the weak market. By 2011–2013, three years' worth of graduates entered the market simultaneously, creating a severe supply bottleneck.

---

## 🤖 Predictive Modeling Performance
Each model was trained on historical data up to **2023 Q4** and evaluated against a backtest horizon of **2024 Q1 – 2025 Q4**.

### Backtest Visualizations
Here is the visual comparison of forecasts across the different paradigms:

#### London Model Forecasts
- **SARIMAX Model Forecast:** ![SARIMAX London](final_plots/SARIMAX_London.png)
- **Prophet Model Forecast:** ![Prophet London](final_plots/Prophet_Baseline.png)
- **XGBoost Model Forecast:** ![XGBoost London](final_plots/XGBOOST_london.png)

#### North East Model Forecasts
- **SARIMAX Model Forecast:** ![SARIMAX North East](final_plots/SARIMAX_NORTH_EAST.png)
- **Prophet Model Forecast:** ![Prophet North East](final_plots/Prophet_Baseline_North_East.png)
- **XGBoost Model Forecast:** ![XGBoost North East](final_plots/XGBOOST_north_East.png)

### Statistical Benchmarking (MAE & RMSE)

| Region | Model Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :--- | :---: | :---: |
| **London** | SARIMAX | 10.15% | 11.22% |
| | Facebook Prophet | 3.20% | 3.77% |
| | **XGBoost (Winner)** | **3.17%** | **3.77%** |
| **North East** | SARIMAX | 6.30% | 8.34% |
| | Facebook Prophet | 7.04% | 8.35% |
| | **XGBoost (Winner)** | **3.55%** | **3.99%** |

*Analysis:* **XGBoost** consistently outperformed the other models, maintaining an error rate of under 4%. SARIMAX suffered from downward drift, failing to capture the complexity of the macro relationships over the validation horizon.

---

## 🧑‍🤝‍🧑 Policymaker Evaluation: Absolute Human Impact ([human_impact_eval.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/human_impact_eval.py))
To translate abstract model statistics into actionable local government metrics, the RMSE margin of error is applied directly to the active regional youth labor force (Ages 16-24):
$$\text{Human Margin of Error} = \left(\frac{\text{RMSE}}{100}\right) \times \text{Active Youth Labor Force}$$

- **London** (Active Youth Labor Force: ~600,000): An RMSE of 3.77% translates to a forecasting uncertainty of **+/- 22,620** real young individuals.
- **North East** (Active Youth Labor Force: ~150,000): An RMSE of 3.99% translates to a forecasting uncertainty of **+/- 5,985** real young individuals.

![Human Impact Bar Chart](final_plots/human_impact_evaluation_bar.png)

---

## 📊 Feature Importance: Macroeconomic Drivers
The relative importance plots from XGBoost reveal different driver mechanisms across the regions:

| London Macro Drivers | North East Macro Drivers |
| :---: | :---: |
| ![London Drivers](final_plots/Macroeconomic_DRiver_xgboost_London.png) | ![North East Drivers](final_plots/Macroeconomic_DRiver_xgboost_north_East.png) |

- **London** is highly sensitive to national labor demand indicators (`UK_Vacancies_Thousands_Lag_1`) and regional financial output (`GDP_Value_mil_Lag_4`).
- The **North East** is highly path-dependent, showing a heavy reliance on historical target lags (`Youth_Unemployment_Rate_Lag_1`) and inflation shocks (`Inflation_Rate_Lag_4`).

---

## 💻 Streamlit Business Intelligence Dashboard ([app.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/app.py))
The project includes a multi-page interactive dashboard designed to put these insights directly in the hands of regional planners.

### Key BI Features:
- **Interactive Horizon Filtering:** Toggle regions and adjust model horizons.
- **Dynamic KPI Panels:** Real-time updates for latest unemployment rates, vacancy levels, interest rates, and GDP splicing metrics.
- **Horizon Forecasting Charts:** Plotly interactive line charts mapping the models' forecasts against actual validation periods.
- **Interactive Local Time-Aware RAG:** Leverages a local `Ollama` framework running `llama3` to generate council-level risk briefs. By parsing real-time dashboard data, the LLM provides contextual risk analysis and resources mapping advice.

---

## 🚀 Execution & Setup Guide

### 1. Installation
Clone the repository and install all dependencies:
```bash
pip install -r requirements.txt
```
*(See [requirements.txt](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/requirements.txt))*

### 2. Run the Data Pipeline
Recreate the master dataset and engineer features:
```bash
# Standardize and merge datasets
python src/ingestion.py

# Perform GDP splicing, cyclical encoding, and lag matrix construction
python src/features.py
```
*(Scripts: [src/ingestion.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/ingestion.py) and [src/features.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/features.py))*

### 3. Run Analysis & Models
```bash
# Run multicollinearity check
python src/vif_check.py

# Visualize structural breaks and historic shocks
python src/structural_breaks.py

# Fit models and output backtest figures
python src/model_sarimax.py
python src/model_prophet.py
python src/model_xgboost.py

# Calculate policymaker human headcount impact
python src/human_impact_eval.py
```
*(Analysis & Modeling scripts: [src/vif_check.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/vif_check.py), [src/structural_breaks.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/structural_breaks.py), [src/model_sarimax.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/model_sarimax.py), [src/model_prophet.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/model_prophet.py), [src/model_xgboost.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/model_xgboost.py), and [src/human_impact_eval.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/human_impact_eval.py))*

### 4. Launch the Streamlit Dashboard
```bash
streamlit run src/app.py
```
*(BI App: [src/app.py](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/src/app.py) | Ensure [ml_matrix.csv](file:///c:/Users/sharm/OneDrive%20-%20University%20of%20East%20London/COMPLETE%20STUDIES/Dissaration%20(DS7010)/youth_unemployement_dissertation/data/processed/ml_matrix.csv) has been generated in `data/processed/` before launching)*
