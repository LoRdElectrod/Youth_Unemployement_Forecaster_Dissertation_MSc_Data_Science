# pyrefly: ignore [missing-import]
import polars as pl
import pandas as pd
import os

def clean_unemployment(file_path, region_name):
    print(f"[*] Ingesting {region_name} Unemployment Data...")
    df = pl.read_csv(
        file_path, 
        null_values=["*"], 
        columns=["Month ", "Rate_16_24"],
        infer_schema_length=10000
    )
    df = df.rename({"Month ": "Date_String", "Rate_16_24": "Youth_Unemployment_Rate"})
    df = df.filter(
        pl.col("Date_String").str.starts_with("Jan-Mar") |
        pl.col("Date_String").str.starts_with("Apr-Jun") |
        pl.col("Date_String").str.starts_with("Jul-Sep") |
        pl.col("Date_String").str.starts_with("Oct-Dec")
    )
    df = df.with_columns([
        pl.col("Date_String").str.slice(-4).alias("Year"),
        pl.col("Date_String").str.slice(0, 7).alias("Quarter_Prefix")
    ])
    quarter_map = {
        "Jan-Mar": "-03-31", "Apr-Jun": "-06-30", 
        "Jul-Sep": "-09-30", "Oct-Dec": "-12-31"
    }
    df = df.with_columns(pl.col("Quarter_Prefix").replace(quarter_map).alias("Month_Day"))
    df = df.with_columns((pl.col("Year") + pl.col("Month_Day")).str.strptime(pl.Date, "%Y-%m-%d").alias("Date"))
    df = df.with_columns(pl.lit(region_name).alias("Region"))
    df = df.select(["Date", "Region", "Youth_Unemployment_Rate"])
    df = df.drop_nulls("Youth_Unemployment_Rate")
    return df

def process_gdp(file_path):
    print("[*] Ingesting Regional GDP Data...")
    df = pl.read_csv(file_path, infer_schema_length=10000)
    df = df.select([
        pl.col("Region_name").alias("Region"),
        pl.col("Year").cast(pl.Utf8),
        pl.col("GDP_Value_mil").cast(pl.Float64)
    ])
    return df

def process_inflation(file_path):
    print("[*] Ingesting UK Inflation Data...")
    df = pl.read_csv(file_path, infer_schema_length=10000)
    df = df.drop_nulls("CPI Annual Rate %")
    df = df.with_columns([
        pl.col("Year").str.slice(0, 4).alias("Year_Num"),
        pl.col("Year").str.slice(5, 3).alias("Month_Str")
    ])
    month_to_q = {
        "JAN": 1, "FEB": 1, "MAR": 1, "APR": 2, "MAY": 2, "JUN": 2,
        "JUL": 3, "AUG": 3, "SEP": 3, "OCT": 4, "NOV": 4, "DEC": 4
    }
    df = df.with_columns(pl.col("Month_Str").replace(month_to_q).cast(pl.Int32).alias("Quarter"))
    q_to_date = {"1": "-03-31", "2": "-06-30", "3": "-09-30", "4": "-12-31"}
    df = df.with_columns(pl.col("Quarter").cast(pl.Utf8).replace(q_to_date).alias("Month_Day"))
    df = df.with_columns((pl.col("Year_Num") + pl.col("Month_Day")).str.strptime(pl.Date, "%Y-%m-%d").alias("Date"))
    df_quarterly = df.group_by("Date").agg(pl.col("CPI Annual Rate %").mean().alias("Inflation_Rate")).sort("Date")
    return df_quarterly

def process_vacancies(file_path):
    print("[*] Ingesting UK Vacancies Data...")
    df = pl.read_csv(file_path, infer_schema_length=10000)
    df = df.rename({"Total Vacancies (thousands)": "UK_Vacancies_Thousands"})
    
    df = df.filter(
        pl.col("Date").str.starts_with("Jan-Mar") |
        pl.col("Date").str.starts_with("Apr-Jun") |
        pl.col("Date").str.starts_with("Jul-Sep") |
        pl.col("Date").str.starts_with("Oct-Dec")
    )
    df = df.with_columns([
        pl.col("Date").str.slice(-4).alias("Year"),
        pl.col("Date").str.slice(0, 7).alias("Quarter_Prefix")
    ])
    quarter_map = {
        "Jan-Mar": "-03-31", "Apr-Jun": "-06-30", 
        "Jul-Sep": "-09-30", "Oct-Dec": "-12-31"
    }
    df = df.with_columns(pl.col("Quarter_Prefix").replace(quarter_map).alias("Month_Day"))
    df = df.with_columns((pl.col("Year") + pl.col("Month_Day")).str.strptime(pl.Date, "%Y-%m-%d").alias("Date"))
    df = df.select(["Date", "UK_Vacancies_Thousands"]).drop_nulls()
    return df

def process_bank_rate(file_path):
    print("[*] Ingesting Bank of England Rate Data...")
    df = pl.read_csv(file_path, infer_schema_length=10000)
    df = df.with_columns(
        pl.col("Date Changed").str.strptime(pl.Date, "%d %b %y").alias("Bank_Rate_Date")
    ).sort("Bank_Rate_Date")
    
    df = df.select(["Bank_Rate_Date", "Rate"])
    df = df.rename({"Rate": "BoE_Base_Rate"})
    return df

#New Data thing, but with specific sheet name
def process_rti_data(file_path):
    print("[*] Ingesting HMRC RTI Excel Data...")
    # 1. Read sheet 7 skipping the first 6 metadata header rows
    pdf = pd.read_excel(file_path, sheet_name="7. Employees (NUTS1)", header=6)
    
    # 2. Filter target columns
    regions_of_interest = ["London", "North East"]
    pdf = pdf[["Date"] + regions_of_interest].dropna()
    
    # 3. Convert to Polars
    df = pl.from_pandas(pdf)
    
    # 4. Unpivot from wide to long format
    df_long = df.unpivot(
        index="Date",
        on=regions_of_interest,
        variable_name="Region",
        value_name="Payrolled_Employees"
    )
    
    # 5. Parse dates (e.g., "July 2014") and extract Year/Quarter
    df_parsed = df_long.with_columns(
        pl.col("Date").str.strptime(pl.Date, "%B %Y").alias("Date_Parsed")
    ).with_columns(
        pl.col("Date_Parsed").dt.year().alias("Year"),
        pl.col("Date_Parsed").dt.quarter().alias("Quarter")
    )
    
    # 6. Group by Region, Year, Quarter and calculate quarterly mean
    df_quarterly = df_parsed.group_by(["Region", "Year", "Quarter"]).agg(
        pl.col("Payrolled_Employees").mean().alias("RTI_Payrolled_Employees")
    ).sort(["Region", "Year", "Quarter"])
    
    # 7. Map back to quarter-end date
    df_final = df_quarterly.with_columns(
        pl.date(pl.col("Year"), pl.col("Quarter") * 3, 1).dt.month_end().alias("Date")
    ).select(["Date", "Region", "RTI_Payrolled_Employees"])
    
    return df_final

def build_master_dataset():
    print("[-] Building extended master dataset...")
    raw_dir = "data/raw"
    
    # 1. Process all raw data files
    df_lon = clean_unemployment(os.path.join(raw_dir, "unemployement_london.csv"), "London")
    df_ne = clean_unemployment(os.path.join(raw_dir, "unemployement_northeast.csv"), "North East")
    df_unemp = pl.concat([df_lon, df_ne])
    
    df_gdp = process_gdp(os.path.join(raw_dir, "regional_gdp_real.csv"))
    df_inf = process_inflation(os.path.join(raw_dir, "monthly_inflation_data.csv"))
    df_vac = process_vacancies(os.path.join(raw_dir, "VACS01_Vacancies_IN_UK_cleaned_bit.csv"))
    df_rate = process_bank_rate(os.path.join(raw_dir, "Bank Rate history and data  Bank of England Database.csv"))
    
    # Pass the downloaded Excel file name here
    df_rti = process_rti_data(os.path.join(raw_dir, "Earnings and employment from Pay As You Earn Real Time Information.xlsx"))
    
    # 2. Merge Exact Date Matches
    master = df_unemp.join(df_inf, on="Date", how="left")
    master = master.join(df_vac, on="Date", how="left")
    
    # 3. Merge RTI Data
    master = master.join(df_rti, on=["Date", "Region"], how="left")
    
    # 4. Merge GDP Data
    master = master.with_columns(pl.col("Date").dt.year().cast(pl.Utf8).alias("Year"))
    master = master.join(df_gdp, on=["Year", "Region"], how="left")
    master = master.drop("Year")
    
    # 5. Merge Bank Rate (ASOF Join)
    master = master.sort("Date")
    master = master.join_asof(df_rate, left_on="Date", right_on="Bank_Rate_Date", strategy="backward")
    
    # 6. Final Cleanup and Sort
    master = master.sort(["Region", "Date"])
    
    # 7. Save output
    os.makedirs("../data/processed", exist_ok=True)
    master.write_csv("../data/processed/master_dataset.csv")
    print(f"[+] Extended Master dataset saved! Rows: {master.height}")
    print(master.head())

if __name__ == "__main__":
    build_master_dataset()