import os
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm
import boto3
import pystac_client
from loguru import logger
from dotenv import load_dotenv
from eopf.common.constants import OpeningMode
from eopf.common.file_utils import AnyPath
from eopf.store.convert import convert
from datetime import datetime


def setup_logger(log_path, filename_prefix):
    """Setup logger with specified path and prefix"""
    os.makedirs(log_path, exist_ok=True)
    log_filename = f"{log_path}/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.remove()
    logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")
    return log_filename


class S3Connector:
    """A clean connector for S3-compatible storage services"""

    def __init__(self, endpoint_url, access_key_id,
                 secret_access_key, region_name='default'):
        """Initialize the S3Connector with connection parameters"""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name

        logger.info("Initializing S3Connector...")

        # Create session
        self.session = boto3.session.Session()

        # Initialize S3 resource
        self.s3 = self.session.resource(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

        # Initialize S3 client
        self.s3_client = self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

    def get_s3_client(self):
        """Get the boto3 S3 client"""
        return self.s3_client

    def get_s3_resource(self):
        """Get the boto3 S3 resource"""
        return self.s3

    def get_bucket(self, bucket_name):
        """Get a specific bucket by name"""
        return self.s3.Bucket(bucket_name)

    def list_buckets(self):
        """List all available buckets"""
        response = self.s3_client.list_buckets()
        if 'Buckets' in response:
            return [bucket['Name'] for bucket in response['Buckets']]
        return []


def prepare_paths(path_dir):
    logger.info(f"Preparing paths from directory: {path_dir}")

    df_input = pd.read_csv(f"{path_dir}/input.csv")
    df_output = pd.read_csv(f"{path_dir}/target.csv")

    df_input["path"] = df_input["Name"].apply(lambda x: os.path.join(path_dir, "input", os.path.basename(x).replace(".SAFE", "")))
    df_output["path"] = df_output["Name"].apply(lambda x: os.path.join(path_dir, "target", os.path.basename(x).replace(".SAFE", "")))
    df_input["S3Path"] = df_input["S3Path"].apply(lambda x: f"s3:/{x}" if not x.startswith("s3://") else x)
    df_output["S3Path"] = df_output["S3Path"].apply(lambda x: f"s3:/{x}" if not x.startswith("s3://") else x)

    logger.info(f"Paths prepared: {len(df_input)} input files, {len(df_output)} target files.")
    return df_input, df_output


def download_sentinel_data(df_input, df_output, base_dir):
    """Download Sentinel data from S3 to local directories"""
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Created local directories: {input_dir}, {output_dir}")

    S3_CONFIG = {
        "key": ACCESS_KEY_ID,
        "secret": SECRET_ACCESS_KEY,
        "client_kwargs": {
            "endpoint_url": ENDPOINT_URL,
            "region_name": "default"
        }
    }

    target_store_config = dict(mode=OpeningMode.CREATE_OVERWRITE)

    logger.info("Starting input file downloads...")
    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Input files"):
        try:
            product_url = row['S3Path']
            zarr_filename = os.path.basename(product_url).replace('.SAFE', '.zarr')
            zarr_path = os.path.join(input_dir, zarr_filename)

            logger.info(f"Downloading input: {product_url} -> {zarr_path}")
            convert(AnyPath(product_url, **S3_CONFIG), zarr_path, target_store_kwargs=target_store_config)
        except Exception as e:
            logger.error(f"Error downloading {product_url}: {str(e)}")

    logger.info("Starting target file downloads...")
    for _, row in tqdm(df_output.iterrows(), total=len(df_output), desc="Target files"):
        try:
            product_url = row['S3Path']
            zarr_filename = os.path.basename(product_url).replace('.SAFE', '.zarr')
            zarr_path = os.path.join(output_dir, zarr_filename)

            logger.info(f"Downloading target: {product_url} -> {zarr_path}")
            convert(AnyPath(product_url, **S3_CONFIG), zarr_path, target_store_kwargs=target_store_config)
        except Exception as e:
            logger.error(f"Error downloading {product_url}: {str(e)}")


# Main execution
if __name__ == "__main__":
    load_dotenv()
    version = "V4"
    DATASET_DIR = f"/mnt/disk/dataset/sentinel-ai-processor/{version}"
    TRAIN_DIR = f"{DATASET_DIR}/train/"
    VAL_DIR = f"{DATASET_DIR}/val/"
    TEST_DIR = f"{DATASET_DIR}/test/"
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")
    ENDPOINT_URL = 'https://eodata.dataspace.copernicus.eu'
    ENDPOINT_STAC = "https://stac.dataspace.copernicus.eu/v1/"
    BUCKET_NAME = "eodata"

    if not ACCESS_KEY_ID or not SECRET_ACCESS_KEY:
        logger.error("Missing ACCESS_KEY_ID or SECRET_ACCESS_KEY in environment variables.")
        exit(1)

    setup_logger(DATASET_DIR, "sentinel_query_log")

    logger.info("Connecting to STAC and S3 services...")
    catalog = pystac_client.Client.open(ENDPOINT_STAC)

    connector = S3Connector(
        endpoint_url=ENDPOINT_URL,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        region_name='default'
    )

    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()
    buckets = connector.list_buckets()
    logger.info(f"Available buckets: {buckets}")



    logger.info("Preparing paths for all splits...")
    df_train_input, df_train_output = prepare_paths(TRAIN_DIR)
    df_val_input, df_val_output = prepare_paths(VAL_DIR)
    df_test_input, df_test_output = prepare_paths(TEST_DIR)

    logger.info("Starting download process...")
    download_sentinel_data(df_train_input[:140], df_train_output[:140], TRAIN_DIR)
    download_sentinel_data(df_val_input[:40], df_val_output[:40], VAL_DIR)
    download_sentinel_data(df_test_input[:20], df_test_output[:20], TEST_DIR)
    logger.success("All downloads completed!")
