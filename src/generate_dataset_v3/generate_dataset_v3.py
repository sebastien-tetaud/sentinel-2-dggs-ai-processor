import os
import time
import yaml
import shutil
from datetime import datetime, timedelta
import pandas as pd
import requests
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from auth.auth import S3Connector
from utils.utils import remove_last_segment_rsplit
from utils.cdse_utils import (create_cdse_query_url, download_bands)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def save_config_copy(config, config_path, dataset_dir):
    """Save a copy of the config file to the dataset directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_filename = os.path.basename(config_path)
    config_name, config_ext = os.path.splitext(config_filename)
    target_path = os.path.join(dataset_dir, f"{config_name}_{timestamp}{config_ext}")

    # Save a copy of the config to the dataset directory
    with open(target_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    logger.info(f"Saved configuration copy to {target_path}")
    return target_path


def setup_environment(config):
    """Set up environment variables and directories for the dataset"""
    # Keep these from environment variables
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

    # Get other parameters from config
    ENDPOINT_URL = config['endpoint_url']
    ENDPOINT_STAC = config['endpoint_stac']
    BUCKET_NAME = config['bucket_name']
    DATASET_VERSION = config['dataset_version']
    BASE_DIR = config['base_dir']
    DATASET_DIR = f"{BASE_DIR}/{DATASET_VERSION}"
    BANDS = config['bands']

    # Create directories
    input_dir = os.path.join(DATASET_DIR, "input")
    output_dir = os.path.join(DATASET_DIR, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Setup connector
    connector = S3Connector(
        endpoint_url=ENDPOINT_URL,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        region_name='default')

    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()
    bucket = s3.Bucket(BUCKET_NAME)

    return {
        'ENDPOINT_URL': ENDPOINT_URL,
        'ENDPOINT_STAC': ENDPOINT_STAC,
        'BUCKET_NAME': BUCKET_NAME,
        'DATASET_VERSION': DATASET_VERSION,
        'BASE_DIR': BASE_DIR,
        'DATASET_DIR': DATASET_DIR,
        'BANDS': BANDS,
        'input_dir': input_dir,
        'output_dir': output_dir,
        's3': s3,
        's3_client': s3_client,
        'bucket': bucket
    }


def setup_logger(log_path, filename_prefix):
    """Setup logger with specified path and prefix"""
    log_filename = f"{log_path}/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.remove()
    logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")
    return log_filename


def query_sentinel_data(bbox, start_date, end_date, max_items, max_cloud_cover):
    """Query Sentinel data for the specified parameters"""
    # Generate the polygon string from bbox [minx, miny, maxx, maxy]
    polygon = f"POLYGON (({bbox[0]} {bbox[1]}, {bbox[0]} {bbox[3]}, {bbox[2]} {bbox[3]}, {bbox[2]} {bbox[1]}, {bbox[0]} {bbox[1]}))"

    # Initialize empty lists to store all results
    all_l1c_results = []
    all_l2a_results = []

    # Loop through the date range with a step of 10 days
    current_date = start_date
    while current_date < end_date:
        # Calculate the end of the current interval
        next_date = min(current_date + timedelta(days=10), end_date)

        # Format the dates for the OData query
        start_interval = f"{current_date.strftime('%Y-%m-%dT00:00:00.000Z')}"
        end_interval = f"{next_date.strftime('%Y-%m-%dT23:59:59.999Z')}"

        date_interval = f"{current_date.strftime('%Y-%m-%d')}/{next_date.strftime('%Y-%m-%d')}"

        try:
            # Query for L2A products
            l2a_query_url = create_cdse_query_url(
                product_type="MSIL2A",
                polygon=polygon,
                start_interval=start_interval,
                end_interval=end_interval,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
                orderby="ContentDate/Start"
            )
            l2a_json = requests.get(l2a_query_url).json()
            l2a_results = l2a_json.get('value', [])

            # Add interval metadata
            for item in l2a_results:
                item['query_interval'] = date_interval

            # Query for L1C products
            l1c_query_url = create_cdse_query_url(
                product_type="MSIL1C",
                polygon=polygon,
                start_interval=start_interval,
                end_interval=end_interval,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
                orderby="ContentDate/Start"
            )
            l1c_json = requests.get(l1c_query_url).json()
            l1c_results = l1c_json.get('value', [])

            # Add interval metadata
            for item in l1c_results:
                item['query_interval'] = date_interval

            # Log counts
            l1c_count = len(l1c_results)
            l2a_count = len(l2a_results)

            if l1c_count != l2a_count:
                logger.warning(f"Mismatch in counts for {date_interval}: L1C={l1c_count}, L2A={l2a_count}")

            # Append results
            all_l1c_results.extend(l1c_results)
            all_l2a_results.extend(l2a_results)

            logger.info(f"L1C Items for {date_interval}: {l1c_count}")
            logger.info(f"L2A Items for {date_interval}: {l2a_count}")
            logger.info("####")

        except Exception as e:
            logger.error(f"Error processing interval {date_interval}: {str(e)}")

        # Move to the next interval
        current_date = next_date

    return all_l1c_results, all_l2a_results


def queries_curation(all_l1c_results, all_l2a_results):
    """Process and align L1C and L2A data to ensure they match"""
    # Create DataFrames
    df_l1c = pd.DataFrame(all_l1c_results)
    df_l2a = pd.DataFrame(all_l2a_results)

    # Select required columns
    df_l2a = df_l2a[["Name", "S3Path", "Footprint", "GeoFootprint", "Attributes"]]
    df_l1c = df_l1c[["Name", "S3Path", "Footprint", "GeoFootprint", "Attributes"]]

    # Extract cloud cover
    df_l1c['cloud_cover'] = df_l1c['Attributes'].apply(lambda x: x[2]["Value"])
    df_l2a['cloud_cover'] = df_l2a['Attributes'].apply(lambda x: x[2]["Value"])
    # Drop the Attributes column (note: inplace=True needed or need to reassign)
    df_l1c = df_l1c.drop(columns=['Attributes'], axis=1)
    df_l2a = df_l2a.drop(columns=['Attributes'], axis=1)
    # Create id_key for matching
    df_l2a['id_key'] = df_l2a['Name'].apply(remove_last_segment_rsplit)
    df_l2a['id_key'] = df_l2a['id_key'].str.replace('MSIL2A_', 'MSIL1C_')
    df_l1c['id_key'] = df_l1c['Name'].apply(remove_last_segment_rsplit)

    # Remove duplicates
    df_l2a = df_l2a.drop_duplicates(subset='id_key', keep='first')
    df_l1c = df_l1c.drop_duplicates(subset='id_key', keep='first')

    # Align both datasets
    df_l2a = df_l2a[df_l2a['id_key'].isin(df_l1c['id_key'])]
    df_l1c = df_l1c[df_l1c['id_key'].isin(df_l2a['id_key'])]

    # Make sure the order is the same
    df_l2a = df_l2a.set_index('id_key')
    df_l1c = df_l1c.set_index('id_key')

    df_l2a = df_l2a.loc[df_l1c.index].reset_index()
    df_l1c = df_l1c.reset_index()

    return df_l1c, df_l2a


def validate_data_alignment(df_l1c, df_l2a):
    """Validate that the data is properly aligned"""
    mismatches = 0
    for i in range(min(len(df_l1c), len(df_l2a))):
        if df_l1c['id_key'][i] != df_l2a['id_key'][i]:
            logger.error(f"Mismatch: {df_l1c['id_key'][i]} != {df_l2a['id_key'][i]}")
            mismatches += 1

    if mismatches == 0:
        logger.info(f"All {len(df_l1c)} records are properly aligned")
    else:
        logger.warning(f"Found {mismatches} mismatches in data alignment")


def main():
    config_path = 'cfg/config_dataset.yaml'

    # Load configuration from YAML
    config = load_config(config_path)

    # Initialize environment using config
    env = setup_environment(config)

    # Save a copy of the config file to the dataset directory
    saved_config_path = save_config_copy(config, config_path, env['DATASET_DIR'])

    # Get query parameters from config
    query_config = config['query']
    bbox = query_config['bbox']
    start_date = datetime.strptime(query_config['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(query_config['end_date'], '%Y-%m-%d')
    max_items = query_config['max_items']
    max_cloud_cover = query_config['max_cloud_cover']

    # Set up logger for query
    setup_logger(env['DATASET_DIR'], "sentinel_query_log")

    # Log query parameters
    logger.info(f"Using configuration from: {saved_config_path}")
    logger.info(f"Query parameters:")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Max items per request: {max_items}")
    logger.info(f"Max cloud cover: {max_cloud_cover}%")

    # Query Sentinel data
    all_l1c_results, all_l2a_results = query_sentinel_data(
        bbox, start_date, end_date, max_items, max_cloud_cover
    )

    # Process and align data
    df_l1c, df_l2a = queries_curation(all_l1c_results, all_l2a_results)

    # Save full datasets
    df_l1c.to_csv(f"{env['DATASET_DIR']}/input_l1c.csv")
    df_l2a.to_csv(f"{env['DATASET_DIR']}/output_l2a.csv")

    # Validate alignment
    validate_data_alignment(df_l1c, df_l2a)

    # Set up logger for download
    setup_logger(env['DATASET_DIR'], "sentinel_download_log")

    # Download the data
    # download_sentinel_data(
    #     s3_client=env['s3_client'],
    #     bucket_name=env['BUCKET_NAME'],
    #     df_l1c=df_l1c,
    #     df_l2a=df_l2a,
    #     bands=env['BANDS'],
    #     input_dir=env['input_dir'],
    #     output_dir=env['output_dir'],
    #     download_config=config['download']
    # )

if __name__ == "__main__":
    main()
