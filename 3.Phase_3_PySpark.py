# PYSPARK FOR DATA ENGINEERING

# SETUP

from pyspark.sql import SparkSession
import os

home = os.path.expanduser("~")
project_root = os.path.join(home, "python_for_de")

# Create Spark session
spark = SparkSession.builder \
    .appName("PySpark-DE-Phase3") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

print("✅ Spark Session Created Successfully!")

# CREATE DATAFRAMES

data = [
    (1, "Arjun", "Chennai", 70000),
    (2, "Gokul", "Bangalore", 85000),
    (3, "Pritha", "Hyderabad", 60000),
    (4, "Krithi", "Chennai", 95000)
]
columns = ["ID", "Name", "City", "Salary"]

df = spark.createDataFrame(data, columns)
df.show()

df2 = spark.read.csv(f"file://{project_root}/input/employees.csv", header=True, inferSchema=True)
df2.show()

# SCHEMA MANAGEMENT

from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("ID", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("City", StringType(), True),
    StructField("Salary", IntegerType(), True)
])

df3 = spark.createDataFrame(data, schema)
df3.printSchema()

# DATA CLEANING

from pyspark.sql.functions import col, trim

cleaned_df = df3.dropDuplicates(["ID"]) \
    .na.fill({"City": "Unknown"}) \
    .withColumn("Name", trim(col("Name")))

cleaned_df.show()

# TRANSFORMATIONS

from pyspark.sql.functions import *

# Add new column
df4 = cleaned_df.withColumn("Bonus", col("Salary") * 0.10)

# Filter data
df4 = df4.filter(col("City") == "Chennai")

# Select specific columns
df4.select("Name", "City", "Bonus").show()

# JOINS

dept_data = [
    (1, "Engineering"),
    (2, "HR"),
    (3, "Finance"),
    (4, "Admin")
]
dept_df = spark.createDataFrame(dept_data, ["ID", "Dept"])

joined_df = df3.join(dept_df, "ID", "inner")
joined_df.show()

# Other join types: left, right, outer, cross

# GROUP BY, AGGREGATIONS, SORTING

agg_df = joined_df.groupBy("City").agg(
    count("ID").alias("EmployeeCount"),
    avg("Salary").alias("AvgSalary"),
    sum("Salary").alias("TotalSalary")
)
agg_df.orderBy(col("TotalSalary").desc()).show()

# CONDITIONAL LOGIC

df5 = joined_df.withColumn(
    "Level",
    when(col("Salary") > 80000, "High")
    .when(col("Salary") > 60000, "Medium")
    .otherwise("Low")
)
df5.show()

# WORKING WITH DATES & STRINGS

df6 = df5.withColumn("CurrentDate", current_date()) \
         .withColumn("Year", year(current_date())) \
         .withColumn("NameLength", length(col("Name")))

df6.show()

# USER-DEFINED FUNCTIONS

from pyspark.sql.functions import udf

def categorize_city(city):
    if city == "Chennai":
        return "South"
    elif city == "Bangalore":
        return "South"
    else:
        return "Other"

categorize_udf = udf(categorize_city, StringType())

df7 = df6.withColumn("Region", categorize_udf(col("City")))
df7.show()

# WRITING DATA

df7.write.csv(f"file://{project_root}/output/employees_cleaned.csv", header=True, mode="overwrite")
df7.write.json(f"file://{project_root}/output/employees_cleaned.json", mode="overwrite")
df7.write.parquet(f"file://{project_root}/output/employees_cleaned.parquet", mode="overwrite")

# PARTITIONING & BUCKETING

df7.write.partitionBy("Region").parquet(f"file://{project_root}/output/partitioned_data", mode="overwrite")

# END-TO-END ETL PIPELINE

def etl_pipeline():
    print("\n🚀 Starting PySpark ETL Pipeline...\n")
    
    # Extraction
    data = [
        (1, "Arjun", "Chennai", 70000),
        (2, "Gokul", "Bangalore", 85000),
        (3, "Pritha", "Hyderabad", 60000),
        (4, "Krithi", "Chennai", 95000),
        (5, None, "Pune", 50000)
    ]
    cols = ["ID", "Name", "City", "Salary"]
    df = spark.createDataFrame(data, cols)
    
    # Transformation
    df = df.na.fill({"Name": "Unknown"}) \
             .withColumn("Bonus", col("Salary") * 0.10) \
             .withColumn("Level", when(col("Salary") > 80000, "High").otherwise("Low")) \
             .withColumn("ProcessedAt", current_timestamp())
    
    # Load
    df.write.mode("overwrite").parquet(f"file://{project_root}/output/final_data")
    print("✅ ETL Pipeline Completed Successfully!")

etl_pipeline()