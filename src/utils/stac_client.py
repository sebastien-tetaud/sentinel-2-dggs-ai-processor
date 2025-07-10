import io
from datetime import datetime, timedelta

import random
import requests
from pystac_client import Client
import os

from auth.auth import get_direct_access_token
from utils.image import extract_url_after_filename


def get_product_content(s3_client, bucket_name, object_url):
    """
    Download the content of a product from S3 bucket.

    Args:
        s3_client: boto3 S3 client object
        bucket_name (str): Name of the S3 bucket
        object_url (str): Path to the object within the bucket

    Returns:
        bytes: Content of the downloaded file
    """
    print(f"Downloading {object_url}")

    try:
        # Download the file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_url)
        content = response['Body'].read()
        print(f"Successfully downloaded {object_url}")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        raise

    return content


def get_product(s3_resource, bucket_name, object_url, output_path):
    """
    Download a product from S3 bucket and create output directory if it doesn't exist.

    Args:
        s3_resource: boto3 S3 resource object
        bucket_name (str): Name of the S3 bucket
        object_url (str): Path to the object within the bucket
        output_path (str): Local directory to save the file

    Returns:
        str: Path to the downloaded file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Extract filename from the object URL
    _, filename = os.path.split(object_url)

    # Full path where the file will be saved
    local_file_path = os.path.join(output_path, filename)

    print(f"Downloading {object_url} to {local_file_path}...")

    try:
        # Download the file from S3
        s3_resource.Bucket(bucket_name).download_file(object_url, local_file_path)
        print(f"Successfully downloaded to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        raise

    return local_file_path


def download_sentinel_image(username, password, start_date, end_date,
                            bbox=[-180, -90, 180, 90], limit=10):
    """
    Download a random Sentinel-2 image based on criteria.

    Args:
        username (str): DESTINE username
        password (str): DESTINE password
        # date_range (str): Date range in format "YYYY-MM-DD/YYYY-MM-DD"
        cloud_cover (int, optional): Maximum cloud cover percentage
        bbox (list): Bounding box coordinates [west, south, east, north]
        limit (int): Maximum number of results to return

    Returns:
        tuple: (image_content or error_message, metadata)
    """
    # Get access token
    token_result = get_direct_access_token(username=username, password=password)
    if not token_result:
        return "Failed to authenticate", None

    access_token = token_result["access_token"]

    # Set up STAC API client
    stac_base_url = "https://cachea.destine.eu"
    stac_url = f"{stac_base_url}/stac/api"
    catalog = Client.open(stac_url)

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    days_between = (end_date - start_date).days
    random_start_day = random.randint(0, days_between - 7)  # Ensure we have 7 days
    random_start_date = start_date + timedelta(days=random_start_day)
    random_end_date = random_start_date + timedelta(days=1)

    # Format dates for the API
    start_date_str = random_start_date.strftime("%Y-%m-%d")
    end_date_str = random_end_date.strftime("%Y-%m-%d")

    # Build search parameters
    search_params = {
        "method": "GET",
        "collections": ["SENTINEL-2"],
        "bbox": bbox,
        "datetime": f"{start_date_str}/{end_date_str}",
        "limit": limit
    }


    # Search for Sentinel-2 images
    search = catalog.search(**search_params)

    # Get a list of items
    items = list(search.items())
    if not items:
        return "No Sentinel-2 images found", None

    # Select a random item
    random_item = random.choice(items)

    # Get metadata for the selected item
    metadata = {
        "id": random_item.id,
        "datetime": random_item.datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "bbox": random_item.bbox,
    }


    # Get the assets of the random item
    assets = random_item.assets
    asset_keys = list(assets.keys())

    # Filter the assets to get the one that ends with *_TCI_60m.jp2
    tci_assets = [assets[key].href for key in asset_keys if assets[key].href.endswith('_TCI_60m.jp2')]

    if not tci_assets:
        return "No TCI assets found in the selected image", None

    filepath = extract_url_after_filename(tci_assets[0])
    metadata["filename"] = os.path.basename(filepath)

    # Download the file
    url = f"{stac_base_url}/stac/download?filename={filepath}"

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.post(url, headers=headers, data={})

    if response.status_code == 200:
        return response.content, metadata
    else:
        return f"Failed to download the file. Status code: {response.status_code}", None


def create_cdse_query_url(
    collection_name="SENTINEL-2",
    product_type="MSIL2A",
    polygon=None,
    start_interval=None,
    end_interval=None,
    max_cloud_cover=100,
    max_items=1000,
    additional_filters=None,
    orderby="ContentDate/Start"  # Add orderby parameter with default value
):
    """
    Create a query URL for the Copernicus Data Space Ecosystem OData API.

    Parameters:
    -----------
    collection_name : str
        The collection name (e.g., 'SENTINEL-2', 'SENTINEL-1')
    product_type : str
        The product type (e.g., 'MSIL2A', 'MSIL1C', 'GRD')
    polygon : str
        WKT polygon string for spatial filtering
    start_interval : str
        Start time in ISO format with Z for UTC (e.g., '2023-01-01T00:00:00.000Z')
    end_interval : str
        End time in ISO format with Z for UTC (e.g., '2023-01-31T23:59:59.999Z')
    max_cloud_cover : int
        Maximum cloud cover percentage (0-100)
    max_items : int
        Maximum number of items to return
    additional_filters : list
        List of additional filter strings to add to the query
    orderby : str or None
        Field to order results by (e.g., 'ContentDate/Start', 'ContentDate/Start desc')
        Set to None to skip ordering

    Returns:
    --------
    str
        Complete URL for the OData API query
    """

    # Basic filter for collection
    filter_parts = [f"Collection/Name eq '{collection_name}'"]

    # Add spatial filter if provided
    if polygon:
        filter_parts.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')")

    # Add product type filter
    if product_type:
        filter_parts.append(f"contains(Name,'{product_type}')")

    # Add temporal filters if provided
    if start_interval:
        filter_parts.append(f"ContentDate/Start gt {start_interval}")
    if end_interval:
        filter_parts.append(f"ContentDate/Start lt {end_interval}")

    # Add cloud cover filter if applicable
    # Only add for optical sensors (Sentinel-2)
    if collection_name == 'SENTINEL-2' and max_cloud_cover < 100:
        filter_parts.append(
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
            f"att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
        )

    # Add any additional filters
    if additional_filters:
        filter_parts.extend(additional_filters)

    # Construct the URL with all filters
    filter_string = " and ".join(filter_parts)
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter={filter_string}"

    # Add orderby parameter if specified
    if orderby:
        url += f"&$orderby={orderby}"

    # Add top parameter for limiting results
    url += f"&$top={max_items}"

    url += "&$expand=Attributes"

    return url