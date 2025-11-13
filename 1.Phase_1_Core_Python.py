# CORE PYTHON

# 1. LIST - BASICS

# --- Basic Lists ---
nums = [10, 20, 30, 40, 50]
print("Original:", nums)

# Common operations
nums.append(60)                   # Add item
nums.insert(0, 5)                 # Insert at start
nums.remove(30)                   # Remove element
nums.reverse()                    # Reverse list
print("After modifications:", nums)

# Sorting
sorted_nums = sorted(nums)
print("Sorted:", sorted_nums)

# Aggregate functions
print("Sum:", sum(nums))
print("Max:", max(nums))
print("Min:", min(nums))

# LIST - COMPREHENSIONS

nums = [5, 12, 7, 20, 3]

# Transform
squares = [x**2 for x in nums]
# Filter
filtered = [x for x in nums if x > 10]
# Transform + Filter
even_doubles = [x*2 for x in nums if x % 2 == 0]
# Flatten nested lists
flattened = [x for row in [[1, 2], [3, 4]] for x in row]

print("Squares:", squares)
print("Filtered:", filtered)
print("Even Doubles:", even_doubles)
print("Flattened:", flattened)

# DICTIONARY BASICS

# --- Dictionaries ---
prices = {"apple": 120, "banana": 50, "mango": 80}

# Access & Modify
print(prices["apple"])
prices["grape"] = 100
print("After adding:", prices)

# Safe get
print("Watermelon price:", prices.get("watermelon", "Not available"))

# Looping
for fruit, price in prices.items():
    print(f"{fruit}: ₹{price}")

# Dictionary Comprehension
discount = {k: v*0.9 for k, v in prices.items()}
expensive = {k: v for k, v in prices.items() if v > 70}
print("Discounted:", discount)
print("Expensive only:", expensive)


# 2. FILE HANDLING

# CSV HANDLING

import csv

# --- Create a CSV file for demo ---
data = [
    {"Name": "Arjun", "Age": 25, "City": "Chennai"},
    {"Name": "Gokul", "Age": 26, "City": "Bangalore"},
    {"Name": "Pritha", "Age": 27, "City": "Hyderabad"}
]

with open('input/people.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["Name", "Age", "City"])
    writer.writeheader()
    writer.writerows(data)
print("CSV file created successfully ✅")

# --- Read the CSV file ---
with open('input/people.csv', 'r') as f:
    reader = csv.DictReader(f)
    print("\nCSV Read Result:")
    for row in reader:
        print(row["Name"], "-", row["City"])

# JSON HANDLING

import json

# --- Create JSON file ---
data_json = [
    {"Name": "Alice", "Age": 24, "Country": "USA"},
    {"Name": "Bob", "Age": 28, "Country": "India"}
]

with open('input/data.json', 'w') as f:
    json.dump(data_json, f, indent=4)
print("\nJSON file created successfully ✅")

# --- Read JSON file ---
with open('input/data.json', 'r') as f:
    content = json.load(f)
print("JSON Read Result:", content)

# --- Convert JSON string ---
json_str = '{"Name": "Krithi", "Role": "Lawyer"}'
parsed = json.loads(json_str)
print("Converted from JSON string:", parsed["Name"])

# TXT FILE HANDLING

# --- Write to TXT file ---
with open('input/notes.txt', 'w') as f:
    f.write("Hello Arjun!\n")
    f.write("This is your DE prep notes file.\n")

# --- Read the TXT file ---
with open('input/notes.txt', 'r') as f:
    lines = f.readlines()
print("\nTXT Read Result:", lines)

# 3. ERROR HANDLING

try:
    with open('missing.csv', 'r') as f:
        data = f.read()
except FileNotFoundError:
    print("\n⚠️ File not found! Check your path.")
except Exception as e:
    print("Unexpected Error:", e)
finally:
    print("File handling process complete ✅")

# 4. FUNCTIONS & REUSABILITY

# --- Simple Function ---
def calculate_total(price, qty):
    return price * qty

print("\nFunction Example:")
print("Total:", calculate_total(250, 3))

# --- Working with Mutable vs Immutable ---
def mutate_list(lst):
    lst.append("new")
    lst[0] = "changed"

items = ["orig"]
mutate_list(items)
print("Mutated list:", items)


# 5. MODULARIZATION EXAMPLE

# --- utils.py ---
def clean_df(df):
    df.dropna(inplace=True)
    df["Total"] = df["Price"] * df["Qty"]
    return df

# --- main.py ---
from utils import clean_df
import pandas as pd

data = {
    "Item": ["Pen", "Book", "Pencil"],
    "Price": [10, 40, 5],
    "Qty": [2, 1, 5]
}
df = pd.DataFrame(data)

df = clean_df(df)
df.to_csv("output/out.csv", index=False)
print("\nData cleaned & exported successfully ✅")

# 6. MINI ETL

import csv, time, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def etl_process(input_file, output_file):
    start = time.time()
    try:
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]

        # Transformation
        for r in rows:
            r["Total"] = float(r["Age"]) * 1.5

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Name", "Age", "City", "Total"])
            writer.writeheader()
            writer.writerows(rows)

        logging.info("ETL Completed Successfully ✅")
        logging.info(f"Time Taken: {round(time.time() - start, 2)} sec")
    except Exception as e:
        logging.error("ETL Failed: %s", e)

etl_process("input/people.csv", "output/transformed.csv")


