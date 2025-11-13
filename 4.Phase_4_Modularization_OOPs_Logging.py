# MODULARIZATION

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

# main.py
from utils import extract_csv, transform_data, load_to_csv

def run_etl():
    raw_data = extract_csv('data_file/raw_sales.csv')
    transformed = transform_data(raw_data)
    load_to_csv(transformed, 'data_file/processed_sales.csv')

if __name__ == "__main__":
    run_etl()

# LOGGING SETUP

# logger.py
import logging

def setup_logger(log_file='etl.log'):
    logger = logging.getLogger('ETL_LOGGER')
    logger.setLevel(logging.INFO)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add handler
    logger.addHandler(fh)
    return logger

# main.py
from utils import extract_csv, transform_data, load_to_csv
from logger import setup_logger

logger = setup_logger()

def run_etl():
    try:
        logger.info("ETL Started")
        raw_data = extract_csv('data_file/raw_sales.csv')
        transformed = transform_data(raw_data)
        load_to_csv(transformed, 'data_file/processed_sales.csv')
        logger.info("ETL Completed Successfully ✅")
    except Exception as e:
        logger.error(f"ETL Failed ❌ | Error: {e}")

if __name__ == "__main__":
    run_etl()

# OOPS for ETL

# etl_pipeline.py
import pandas as pd
from logger import setup_logger

class ETLPipeline:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.logger = setup_logger()

    def extract(self):
        self.logger.info(f"Extracting data from {self.input_path}")
        self.df = pd.read_csv(self.input_path)

    def transform(self):
        self.logger.info("Transforming data...")
        self.df['Total'] = self.df['Quantity'] * self.df['Price']
        self.df['Category'] = self.df['Category'].str.title()

    def load(self):
        self.logger.info(f"Loading data to {self.output_path}")
        self.df.to_csv(self.output_path, index=False)
        self.logger.info("Load Complete ✅")

    def run(self):
        try:
            self.logger.info("Starting ETL Pipeline")
            self.extract()
            self.transform()
            self.load()
            self.logger.info("ETL Pipeline Completed Successfully ✅")
        except Exception as e:
            self.logger.error(f"ETL Pipeline Failed ❌ | {e}")

# main.py
from etl_pipeline import ETLPipeline

if __name__ == "__main__":
    etl = ETLPipeline(
        input_path='data_file/raw_sales.csv',
        output_path='data_file/processed_sales.csv'
    )
    etl.run()

