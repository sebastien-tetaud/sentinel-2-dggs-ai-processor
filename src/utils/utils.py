import yaml
from urllib.parse import urlparse
import pandas as pd
import os

def load_config(config_path="cfg/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def extract_s3_path_from_url(url):
    """
    Extracts the S3 object path from an S3 URL or URI.

    This function parses S3 URLs/URIs and returns just the object path portion,
    removing the protocol (s3://), bucket name, and any leading slashes.

    Args:
        url (str): The full S3 URI (e.g., 's3://eodata/path/to/file.jp2')

    Returns:
        str: The S3 object path (without protocol, bucket name and leading slashes)
    """
    # If it's not an S3 URI, return it unchanged
    if not url.startswith('s3://'):
        return url

    # Parse the S3 URI
    parsed_url = urlparse(url)

    # Ensure this is an S3 URL
    if parsed_url.scheme != 's3':
        raise ValueError(f"URL {url} is not an S3 URL")

    # Extract the path without leading slashes
    object_path = parsed_url.path.lstrip('/')

    return object_path


def prepare_paths(path_dir):
    """
    Prepare paths for input and output datasets from CSV files.

    Args:
        path_dir (str): Directory containing input and target CSV files.

    Returns:
        DataFrame, DataFrame: Two DataFrames for input and output datasets.
    """
    df_input = pd.read_csv(f"{path_dir}/input.csv")
    df_output = pd.read_csv(f"{path_dir}/target.csv")

    df_input["path"] = df_input["Name"].apply(
        lambda x: os.path.join(path_dir, "input", os.path.basename(x).replace(".SAFE", ""))
    )
    df_output["path"] = df_output["Name"].apply(
        lambda x: os.path.join(path_dir, "target", os.path.basename(x).replace(".SAFE", ""))
    )

    return df_input, df_output


def remove_last_segment_rsplit(sentinel_id):
    # Split from the right side, max 1 split
    parts = sentinel_id.rsplit('_', 1)
    return parts[0]