# utils.py
import pandas as pd

def extract_csv(file_path):
    print("🔹 Extracting data from:", file_path)
    return pd.read_csv(file_path)

def transform_data(df):
    print("🔹 Transforming data...")
    df['Total'] = df['Quantity'] * df['Price']
    df['Category'] = df['Category'].str.upper()
    return df

def load_to_csv(df, output_path):
    print("🔹 Loading data to:", output_path)
    df.to_csv(output_path, index=False)
    print("✅ Load complete.")

def clean_df(df):
    df.dropna(inplace=True)
    df["Total"] = df["Price"] * df["Qty"]
    return df