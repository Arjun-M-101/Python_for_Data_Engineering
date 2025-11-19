# PANDAS FOR DATA ENGINEERING

'''
📖 Why Pandas Matters in DE

- Foundation Layer: Core tool for tabular data processing and rapid data pipeline prototyping.
- Versatility Layer: Handles CSV, JSON, Excel, SQL, Parquet and bulk data in/out easily.
- Bridge Layer: Helps transition from POC to production pipelines (often with Spark).

Scope for DE:

- Data cleaning and shaping (nulls/outliers, schema fixes).
- Lightweight ETL for files, APIs, test batches, PoCs.
- Exploratory validation before big data migration.
'''

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

'''
📖 Pandas DE‑Focused Patterns

- Schema-Aware Reads: Use pd.read_csv(), read_json(), etc., specifying dtype, parse_dates, and custom delimiters for precise ingestion.
- Null Handling: df.dropna(), df.fillna() for quick bulk cleaning.
- Feature Engineering: df['New'] = ..., .apply(), .map(), .assign() for new columns/transformations.
- Joins/Merges: pd.merge(left, right, on=col, how='left') — combine multiple datasets cleanly.
- Groupby-Aggregations: df.groupby(col).agg() for quick pivots, summaries.
- Export: df.to_csv(), .to_json(), .to_excel() for downstream use.
- Pipeline Chaining: Leverage method chaining (df.pipe(...)) for readable, step-wise ETL flows.
'''

'''
NUMPY

NumPy for Data Engineering — The Complete Field Manual.  
Think of it as a *cinematic training montage* for your DE toolkit.  

---

# NumPy for Data Engineering — Field Manual

## 📖 Why NumPy Matters in DE
- Foundation Layer: Pandas, scikit‑learn, and many ETL/ML tools are built on NumPy arrays.
- Speed Layer: Vectorized operations run in C under the hood — far faster than Python loops.
- Bridge Layer: Lets you translate Data Science prototypes into DE‑ready transformations.
- Scope for DE:  
  - Quick in‑memory transformations before ingestion.  
  - Data validation and QA checks.  
  - Pre/post‑processing for APIs, ML models, or analytics.

---

## 📖 Creating Arrays

### From Python Lists'''
import numpy as np

arr = np.array([1, 2, 3])
print(arr, arr.dtype, arr.shape)

### From Ranges
print(np.arange(0, 10, 2))   # [0 2 4 6 8]
print(np.linspace(0, 1, 5))  # [0.   0.25 0.5  0.75 1. ]

### From Zeros/Ones
print(np.zeros((3, 2)))
print(np.ones((2, 4), dtype=int))

#DE Use Case: Initialize placeholder arrays for batch processing or schema‑aligned buffers.

## 📖 Chapter 3 — Shapes & Dimensions
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.ndim)   # 2
print(a.shape)  # (2, 3)
print(a.size)   # 6

#DE Tip: Shape awareness prevents mismatches when merging, reshaping, or exporting.

## 📖 Chapter 4 — Indexing & Slicing
arr = np.array([10, 20, 30, 40, 50])
print(arr[1:4])          # Slice
print(arr[-1])           # Last element
print(arr[[0, 2, 4]])    # Fancy indexing

#Boolean Masking
mask = arr > 25
print(arr[mask])         # [30 40 50]

#DE Use Case: Filter in‑memory datasets before writing to disk.


## 📖 Chapter 5 — Vectorized Operations & Broadcasting
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])

print(x + y)     # [11 22 33]
print(x * 10)    # [10 20 30]

#Broadcasting Example
mat = np.array([[1], [2], [3]])
vec = np.array([10, 20, 30])
print(mat + vec)

#DE Use Case: Apply transformations to entire datasets without loops.

## 📖 Chapter 6 — Aggregations & Reductions
data = np.array([[1, 2, 3], [4, 5, 6]])
print(data.sum(axis=0))  # Column sums
print(data.mean(axis=1)) # Row means

#DE Use Case: Quick QA checks — e.g., row counts, totals, averages.

## 📖 Chapter 7 — Reshaping & Type Casting
mat = np.arange(6).reshape(2, 3)
print(mat.astype(float))

#DE Use Case: Reshape data for APIs, ML models, or binary formats.

## 📖 Chapter 8 — I/O with NumPy
np.savetxt('output.csv', mat, delimiter=',', fmt='%d')
loaded = np.loadtxt('output.csv', delimiter=',')

#DE Use Case: Lightweight save/load for intermediate pipeline steps.

## 📖 Chapter 9 — Problem‑Solving Drills

#1. Filter & Transform
arr = np.array([5, 10, 15, 20, 25])
result = arr[arr > 12] * 2
print(result)  # [30 40 50]

#2. Reshape for Batch Processing
data = np.arange(1, 13)
batches = data.reshape(3, 4)
print(batches)

#3. Quick Validation
data = np.array([[1, 2], [3, 4]])
assert data.shape == (2, 2), "Shape mismatch!"
'''
## 📖 DE‑Focused Patterns
- Pre‑pandas Filtering: Use NumPy masks before DataFrame creation for speed.
- Schema Validation: Check shapes/dtypes before ingestion.
- Vectorized QA: Run column‑wise checks without loops.
- Memory Efficiency: Use dtype to control memory footprint.

---

## 📖 What to Skip (for DE)
- Advanced linear algebra (np.linalg) unless ML‑adjacent.
- Fourier transforms, polynomial fitting, random number generation (unless needed).
- Multi‑dimensional broadcasting beyond 2D unless your pipeline demands it.
'''