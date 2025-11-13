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
