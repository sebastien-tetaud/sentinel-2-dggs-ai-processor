# src/download_sentinel_data.py

import os
import yaml
import pandas as pd
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv

# Import modules
from auth.auth import S3Connector
from utils.cdse_utils import download_bands

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_environment(config):
    """Set up environment variables and directories for the dataset"""
    # Keep these from environment variables
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

    # Get other parameters from config
    ENDPOINT_URL = config['endpoint_url']
    BUCKET_NAME = config['bucket_name']
    DATASET_VERSION = config['dataset_version']
    BASE_DIR = config['base_dir']
    DATASET_DIR = f"{BASE_DIR}/{DATASET_VERSION}"
    BANDS = config['bands']

    # Create directories
    # input_dir = os.path.join(DATASET_DIR, "input")
    # output_dir = os.path.join(DATASET_DIR, "output")
    # os.makedirs(input_dir, exist_ok=True)
    # os.makedirs(output_dir, exist_ok=True)

    # Setup connector
    connector = S3Connector(
        endpoint_url=ENDPOINT_URL,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        region_name='default')

    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()

    return {
        'BUCKET_NAME': BUCKET_NAME,
        'DATASET_DIR': DATASET_DIR,
        'BANDS': BANDS,
        's3_client': s3_client
    }

def setup_logger(log_path, filename_prefix):
    """Setup logger with specified path and prefix"""
    log_filename = f"{log_path}/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.remove()
    logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")
    return log_filename

def download_sentinel_data(s3_client, bucket_name, df_l1c, df_l2a, bands, input_dir, output_dir, download_config):
    """Download the Sentinel data bands"""
    sample_size = download_config.get('sample_size', None)
    resize = download_config.get('resize', False)
    resize_target = download_config.get('resize_target', 1830)
    l1c_resolution = download_config.get('l1c_resolution', None)
    l2a_resolution = download_config.get('l2a_resolution', 60)
    max_attempts = download_config.get('max_attempts', 10)
    retry_delay = download_config.get('retry_delay', 10)

    if sample_size:
        df_l1c_sample = df_l1c.iloc[:sample_size]
        df_l2a_sample = df_l2a.iloc[:sample_size]
    else:
        df_l1c_sample = df_l1c
        df_l2a_sample = df_l2a

    # Download L1C data
    logger.info(f"Downloading {len(df_l1c_sample)} L1C samples...")
    download_bands(
        s3_client=s3_client,
        bucket_name=bucket_name,
        df=df_l1c_sample,
        product_type="L1C",
        bands=bands,
        resize=resize,
        resize_target=resize_target,
        resolution=l1c_resolution,
        output_dir=input_dir,
        max_attempts=max_attempts,
        retry_delay=retry_delay
    )

    # Download L2A data
    logger.info(f"Downloading {len(df_l2a_sample)} L2A samples...")
    download_bands(
        s3_client=s3_client,
        bucket_name=bucket_name,
        df=df_l2a_sample,
        product_type="L2A",
        bands=bands,
        resize=resize,
        resize_target=resize_target,
        resolution=l2a_resolution,
        output_dir=output_dir,
        max_attempts=max_attempts,
        retry_delay=retry_delay
    )

def main():
    # Load environment variables
    load_dotenv()

    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download Sentinel data based on provided config and CSV files')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--l1c-csv', type=str, required=True, help='Path to L1C CSV file')
    parser.add_argument('--l2a-csv', type=str, required=True, help='Path to L2A CSV file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup environment
    env = setup_environment(config)

    # Setup logger
    setup_logger(env['DATASET_DIR'], "sentinel_download_log")

    # Determine CSV file paths
    l1c_csv_path = args.l1c_csv
    l2a_csv_path = args.l2a_csv

    input_dir = os.path.join(env['DATASET_DIR'], "input")
    output_dir = os.path.join(env['DATASET_DIR'], "output")

    # Log info
    logger.info(f"Using configuration from: {args.config}")
    logger.info(f"Loading L1C data from: {l1c_csv_path}")
    logger.info(f"Loading L2A data from: {l2a_csv_path}")

    # Load CSV files
    try:
        df_l1c = pd.read_csv(l1c_csv_path)
        df_l2a = pd.read_csv(l2a_csv_path)

        logger.info(f"Loaded {len(df_l1c)} L1C records and {len(df_l2a)} L2A records")

        # Download data
        download_sentinel_data(
            s3_client=env['s3_client'],
            bucket_name=env['BUCKET_NAME'],
            df_l1c=df_l1c,
            df_l2a=df_l2a,
            bands=env['BANDS'],
            input_dir=input_dir,
            output_dir=output_dir,
            download_config=config['download']
        )

        logger.info("Download completed successfully")

    except Exception as e:
        logger.error(f"Error during download process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
