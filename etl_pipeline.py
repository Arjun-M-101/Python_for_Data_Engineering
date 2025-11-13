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
