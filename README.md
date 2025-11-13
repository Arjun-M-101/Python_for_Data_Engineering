```markdown
# 🐍 Python for Data Engineering

## 🚀 Overview
This repository is a structured learning and practice environment for mastering **Python, Pandas, PySpark, Modularization, Logging, and ETL pipelines** from a Data Engineering perspective.  

It is organized into phases, each focusing on a specific skill set, culminating in **batch & streaming pipeline projects**.

---

## 📂 Folder Structure
```
python_for_data_engineering/
├── 0.Phase_0_Basic_Python.py        # User input/output, data types, loops, functions
├── 1.Phase_1_Core_Python.py         # Collections, comprehensions, file handling, error handling
├── 2.Phase_2_Pandas.py              # Pandas for DE: cleaning, transformations, joins, aggregations
├── 3.Phase_3_PySpark.py             # PySpark for DE: schema, cleaning, joins, aggregations, ETL
├── 4.Phase_4_Modularization_OOPs_Logging.py # Modular ETL with utils + logging
├── 5.Phase_5_Projects.py            # Batch & streaming pipeline projects (reference)
├── utils.py                         # Helper functions for ETL (extract, transform, load)
├── logger.py                        # Logging setup for ETL pipelines
├── etl_pipeline.py                  # OOP-based ETL pipeline class
├── requirements.txt                 # Python dependencies
├── etl.log                          # Log file generated during ETL runs
├── input/                           # Input datasets (CSV, JSON, TXT)
│   ├── people.csv
│   ├── employees.csv
│   ├── data.json
│   └── notes.txt
├── data_file/                       # Raw & processed sales data
│   ├── raw_sales.csv
│   └── processed_sales.csv
├── output/                          # ETL outputs (CSV, JSON, Excel, Parquet)
│   ├── final_sales_data.csv/json/xlsx
│   ├── employees_cleaned.csv/json/parquet
│   ├── processed_people.csv
│   ├── transformed.csv
│   ├── final_data/                  # Parquet outputs
│   └── partitioned_data/            # Partitioned parquet outputs by Region
└── README.md                        # Project documentation
```

---

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Arjun-M-101/Python_for_Data_Engineering.git
   cd python_for_data_engineering
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, pyspark; print('✅ Setup OK')"
   ```

---

## 📖 Phase Breakdown

### Phase 0 – Basic Python
- Input/output, conditionals, loops
- Functions, mutability, lambda, map/filter/reduce
- Time complexity basics

### Phase 1 – Core Python
- Lists, dicts, comprehensions
- File handling (CSV, JSON, TXT)
- Error handling with try/except/finally
- Modularization example (`utils.py`)

### Phase 2 – Pandas for DE
- Data cleaning (nulls, types, duplicates)
- Transformations (new columns, filtering, feature engineering)
- Joins, groupby, aggregations
- Export to CSV, JSON, Excel
- Mini ETL pipeline with Pandas

### Phase 3 – PySpark for DE
- SparkSession setup
- Schema management
- Cleaning (dropDuplicates, fillna, trim)
- Transformations (withColumn, filter, joins)
- Aggregations, sorting, conditional logic
- UDFs, date/time functions
- Writing outputs (CSV, JSON, Parquet, partitioned)
- End‑to‑end PySpark ETL pipeline

### Phase 4 – Modularization, OOPs, Logging
- `utils.py`: extract, transform, load functions
- `logger.py`: centralized logging setup
- `etl_pipeline.py`: OOP‑based ETL pipeline class

### Phase 5 – Projects
- Batch & streaming pipelines (reference)
- Real‑world DE scenarios

---

## 📦 Outputs
- **CSV/JSON/Excel** exports for cleaned datasets  
- **Parquet** outputs for scalable storage  
- **Partitioned Parquet** by region for analytics  
- **Logs** stored in `etl.log` for monitoring  

---

## ✅ Key Learnings
- Python fundamentals for DE
- Pandas for batch data processing
- PySpark for scalable distributed pipelines
- Modularization & OOP design for ETL
- Logging for observability
- End‑to‑end ETL pipeline implementation

---