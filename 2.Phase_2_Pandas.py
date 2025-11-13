# PANDAS FOR DATA ENGINEERING

# Setup

import pandas as pd
import numpy as np

print("✅ Pandas setup ready! Version:", pd.__version__)


# Datasets

# --- Create Sales Dataset ---
sales_data = {
    "OrderID": [101, 102, 103, 104, 105, 106],
    "CustID": [1, 2, 3, 2, 4, 1],
    "Product": ["Laptop", "Phone", "Tablet", "Phone", "Laptop", "Laptop"],
    "Quantity": [1, 2, np.nan, 1, 3, 2],
    "Price": [70000, 30000, 15000, 30000, 70000, 70000],
    "Date": ["2024-06-12", "2024-06-13", "2024-06-13", "2024-06-14", "2024-06-15", "2024-06-16"]
}
sales_df = pd.DataFrame(sales_data)

# --- Create Customer Dataset ---
customer_data = {
    "CustID": [1, 2, 3, 4],
    "Name": ["Arjun", "Gokul", "Pritha", "Krithi"],
    "City": ["Chennai", "Bangalore", "Hyderabad", "Chennai"],
    "Gender": ["M", "M", "F", "F"]
}
cust_df = pd.DataFrame(customer_data)

print("\n✅ Datasets Created Successfully!")
print("\nSales Data:\n", sales_df)
print("\nCustomer Data:\n", cust_df)

# DATA CLEANING

# --- Check for nulls ---
print("\nNull Values:\n", sales_df.isnull().sum())

# --- Fill missing values ---
sales_df["Quantity"].fillna(1, inplace=True)

# --- Convert data types ---
sales_df["Date"] = pd.to_datetime(sales_df["Date"])

# --- Remove duplicates (if any) ---
sales_df.drop_duplicates(inplace=True)

# --- Add new column (Total Amount) ---
sales_df["Total"] = sales_df["Quantity"] * sales_df["Price"]

print("\nCleaned Data:\n", sales_df)

# TRANSFORMATION & FEATURE ENGINEERING

# --- Extract year, month, day ---
sales_df["Year"] = sales_df["Date"].dt.year
sales_df["Month"] = sales_df["Date"].dt.month_name()

# --- Filter high-value orders ---
high_value = sales_df[sales_df["Total"] > 100000]

# --- Add discount column ---
sales_df["Discount"] = np.where(sales_df["Product"] == "Laptop", 0.1, 0.05)
sales_df["Discounted_Total"] = sales_df["Total"] * (1 - sales_df["Discount"])

print("\nTransformed Data:\n", sales_df)
print("\nHigh Value Orders (>1L):\n", high_value)

# MERGE & JOIN DATASETS

# --- Join (merge) sales with customer info ---
merged_df = pd.merge(sales_df, cust_df, on="CustID", how="left")

print("\nMerged Data (Sales + Customer):\n", merged_df)

# GROUP BY & AGGREGATIONS

# --- Total sales per customer ---
sales_per_cust = merged_df.groupby("Name")["Discounted_Total"].sum().reset_index()
print("\nTotal Sales Per Customer:\n", sales_per_cust)

# --- Total sales per city ---
sales_per_city = merged_df.groupby("City")["Discounted_Total"].sum().reset_index()
print("\nTotal Sales Per City:\n", sales_per_city)

# --- Aggregation with multiple metrics ---
agg_summary = merged_df.groupby("Product").agg(
    Total_Orders=("OrderID", "count"),
    Avg_Price=("Price", "mean"),
    Total_Amount=("Discounted_Total", "sum")
).reset_index()

print("\nAggregated Product Summary:\n", agg_summary)

# SORTING, FILTERING & SLICING

# --- Sort by Total Amount descending ---
top_orders = merged_df.sort_values(by="Discounted_Total", ascending=False)

# --- Filter only Chennai customers ---
chennai_sales = merged_df[merged_df["City"] == "Chennai"]

print("\nTop Orders by Value:\n", top_orders[["Name", "Product", "Discounted_Total"]])
print("\nChennai Sales:\n", chennai_sales)

# LAMBDA & MAP TRANSFORMATIONS

# --- Apply for transformation ---
merged_df["Category"] = merged_df["Product"].apply(lambda x: "Electronics" if x in ["Laptop", "Phone", "Tablet"] else "Others")

# --- Map Gender to Text ---
merged_df["GenderFull"] = merged_df["Gender"].map({"M": "Male", "F": "Female"})

print("\nAfter Apply & Map:\n", merged_df[["Product", "Category", "GenderFull"]])

# EXPORT OUTPUTS TO FILES

merged_df.to_csv("output/final_sales_data.csv", index=False)
merged_df.to_json("output/final_sales_data.json", orient="records", indent=4)
merged_df.to_excel("output/final_sales_data.xlsx", index=False)

print("\n✅ Data Exported Successfully in CSV, JSON & Excel formats!")

# END-TO-END PIPELINE

def mini_pipeline():
    print("\n🚀 Starting Mini ETL Pipeline...\n")
    sales_df = pd.read_csv("input/people.csv")  # pretend this is sales data
    sales_df["Age"] = pd.to_numeric(sales_df["Age"], errors="coerce")
    sales_df.dropna(inplace=True)

    # Transform
    sales_df["Bonus"] = sales_df["Age"].apply(lambda x: 1000 if x > 25 else 500)

    # Add timestamp
    sales_df["ProcessedAt"] = pd.Timestamp.now()

    # Export
    sales_df.to_csv("output/processed_people.csv", index=False)
    print("✅ Pipeline completed & data exported!")

mini_pipeline()