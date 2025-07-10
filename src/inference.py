import io
import os
import sys
import time
import warnings
from functools import wraps

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystac_client
import torch
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

from auth.auth import S3Connector
from model_zoo.models import define_model
from utils.stac_client import get_product_content
from utils.torch import load_model_weights
from utils.utils import extract_s3_path_from_url, load_config

warnings.filterwarnings('ignore')

# Global store for function durations
function_durations = {}

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Store duration in global dictionary
        function_durations[func.__name__] = function_durations.get(func.__name__, []) + [duration]

        logger.info(f"[BENCHMARK] {func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper

@benchmark
def initialize_env(key_id=sys.argv[1], secret_key=sys.argv[2]) -> dict:
    """Load environment variables."""
    try:
        load_dotenv()
        logger.success("Loaded environment variables")
        return {
            "access_key_id": str(key_id),
            "secret_access_key": str(secret_key)
        }
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {}

def remove_last_segment_rsplit(sentinel_id):
    # Split from the right side, max 1 split
    parts = sentinel_id.rsplit('_', 1)
    return parts[0]

@benchmark
def connect_to_s3(endpoint_url: str, access_key_id: str, secret_access_key: str) -> tuple:
    """Connect to S3 storage."""
    try:
        connector = S3Connector(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name='default'
        )
        logger.success(f"Successfully connected to {endpoint_url} ")
        return connector.get_s3_resource(), connector.get_s3_client()
    except Exception as e:
        logger.error(f"Failed to connect to S3 storage: {e}")
        return None, None

@benchmark
def data_query(catalog, bbox: list, start_date: str, end_date: str, max_cloud_cover: int):
    """
    Fetch both L1C and L2A products from CDSE STAC catalog and find matching pairs.

    Args:
        catalog: STAC catalog client
        bbox: Bounding box coordinates [west, south, east, north]
        start_date: Start date in format "YYYY-MM-DD"
        end_date: End date in format "YYYY-MM-DD"
        max_cloud_cover: Maximum cloud cover percentage

    Returns:
        tuple: (matched L1C item, matched L2A item)
    """
    try:
        # Search for L1C products
        logger.info(f"Searching for L1C products from {start_date} to {end_date} in bbox {bbox}")
        l1c_items = catalog.search(
            collections=['sentinel-2-l1c'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            max_items=1000
        ).item_collection()

        # Search for L2A products
        logger.info(f"Searching for L2A products from {start_date} to {end_date} in bbox {bbox}")
        l2a_items = catalog.search(
            collections=['sentinel-2-l2a'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=1000,
        ).item_collection()

        # Filter L2A items to remove those with high nodata percentage
        l2a_items = [item for item in l2a_items if item.properties.get("statistics", {}).get('nodata', 100) < 5]

        # Convert to dataframes for easier matching
        l1c_dicts = [item.to_dict() for item in l1c_items]
        l2a_dicts = [item.to_dict() for item in l2a_items]

        df_l1c = pd.DataFrame(l1c_dicts)
        df_l2a = pd.DataFrame(l2a_dicts)

        if df_l1c.empty or df_l2a.empty:
            logger.warning(f"Found {len(l1c_items)} L1C products and {len(l2a_items)} L2A products (after filtering)")
            return None, None

        logger.info(f"Found {len(l1c_items)} L1C products and {len(l2a_items)} L2A products (after filtering)")

        # Create unique ID keys for matching
        df_l1c['id_key'] = df_l1c['id'].apply(remove_last_segment_rsplit)
        df_l2a['id_key'] = df_l2a['id'].apply(remove_last_segment_rsplit)
        df_l2a['id_key'] = df_l2a['id_key'].str.replace('MSIL2A_', 'MSIL1C_')

        # Remove duplicates
        df_l1c = df_l1c.drop_duplicates(subset='id_key', keep='first')
        df_l2a = df_l2a.drop_duplicates(subset='id_key', keep='first')

        # Find matching items
        df_l2a = df_l2a[df_l2a['id_key'].isin(df_l1c['id_key'])]
        df_l1c = df_l1c[df_l1c['id_key'].isin(df_l2a['id_key'])]

        # Ensure order is aligned
        df_l2a = df_l2a.set_index('id_key')
        df_l1c = df_l1c.set_index('id_key')
        df_l2a = df_l2a.loc[df_l1c.index].reset_index()
        df_l1c = df_l1c.reset_index()

        logger.info(f"Found {len(df_l1c)} matching L1C/L2A pairs")

        if len(df_l1c) == 0:
            return None, None

        # Select a random pair with a fixed seed for reproducibility

        random_idx = np.random.randint(0, len(df_l1c))

        selected_l1c = df_l1c.iloc[random_idx]
        selected_l2a = df_l2a.iloc[random_idx]

        # Convert back to STAC items
        l1c_item = next((item for item in l1c_items if item.id == selected_l1c['id']), None)
        l2a_item = next((item for item in l2a_items if item.id == selected_l2a['id']), None)

        logger.success(f"Selected L1C: {l1c_item.id}, L2A: {l2a_item.id}")
        return l1c_item, l2a_item

    except Exception as e:
        logger.error(f"Error fetching Sentinel data: {e}")
        return None, None

@benchmark
def load_bands_from_s3(s3_client, bucket_name: str, item, bands: list, resize_shape: tuple = (1830, 1830), product_level: str ="L1C") -> np.ndarray:
    """Load bands from S3 storage."""
    try:
        band_data = []
        for band_name in bands:

            if product_level=="L1C":
                logger.info("Loading L1C bands from S3 storage")
                product_url = extract_s3_path_from_url(item.assets[band_name].href)
            else:
                band_name  = f"{band_name}_10m"
                logger.info("Loading L2A bands from S3 storage")
                product_url = extract_s3_path_from_url(item.assets[band_name].href)

            content = get_product_content(s3_client, bucket_name, product_url)
            image = Image.open(io.BytesIO(content)).resize(resize_shape)
            band_data.append(np.array(image))
        logger.success("Loaded bands from S3 storage")
        return np.dstack(band_data)
    except Exception as e:
        logger.error(f"Failed to load bands from S3 storage: {e}")
        return None

def normalize(data_array: np.ndarray) -> tuple:
    """Normalize the data array."""
    try:
        normalized_data, valid_masks = [], []
        for i in range(data_array.shape[2]):
            band = data_array[:, :, i]
            valid_mask = band > 0
            norm_band = band.astype(np.float32)
            norm_band[valid_mask] /= 10000
            norm_band = np.clip(norm_band, 0, 1)
            norm_band[~valid_mask] = 0
            normalized_data.append(norm_band)
            valid_masks.append(valid_mask)
        logger.success("Normalized data array")
        return np.dstack(normalized_data), np.dstack(valid_masks)
    except Exception as e:
        logger.error(f"Failed to normalize data array: {e}")
        return None, None

@benchmark
def preprocess(raw_data: np.ndarray, resize: int, device: torch.device):
    """Preprocess the raw data."""
    try:
        x_data, valid_mask = normalize(raw_data)
        x_data = cv2.resize(x_data, (resize, resize), interpolation=cv2.INTER_AREA)
        valid_mask = cv2.resize(valid_mask.astype(np.uint8), (resize, resize), interpolation=cv2.INTER_NEAREST).astype(bool)
        x_tensor = torch.from_numpy(x_data).float().permute(2, 0, 1).unsqueeze(0).to(device) # [B , C , W, H]
        logger.success("Preprocess raw data successull")
        return x_tensor, valid_mask
    except Exception as e:
        logger.error(f"Failed to preprocess raw data: {e}")
        return None, None

@benchmark
def postprocess(x_tensor: torch.Tensor, pred_tensor: np.ndarray, valid_mask: np.ndarray) -> tuple:
    """Postprocess the prediction."""
    try:
        x_np = x_tensor.cpu().numpy()[0].transpose(1, 2, 0)
        x_np[~valid_mask] = 0.0
        pred_np = pred_tensor
        pred_np[~valid_mask] = 0.0
        # Make sure all values are clipped to 0-1
        x_np = np.clip(x_np, 0, 1)
        pred_np = np.clip(pred_np, 0, 1)
        logger.success("Postprocess model output successull")
        return x_np, pred_np
    except Exception as e:
        logger.error(f"Failed to postprocess model output: {e}")
        return None, None

@benchmark
def load_model(model_cfg: dict, weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load the model."""
    try:
        model = define_model(
            name=model_cfg["model_name"],
            encoder_name=model_cfg["encoder_name"],
            in_channel=model_cfg["in_channel"],
            out_channels=model_cfg["out_channels"],
            activation=model_cfg["activation"]
        )
        model = load_model_weights(model, filename=weights_path)
        logger.success("Model Loaded")
        return model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

@benchmark
def predict(model: torch.nn.Module, x_tensor: torch.Tensor) -> np.ndarray:
    """Make a prediction."""
    try:
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor)
        logger.success("L2A generation successfull")
        return pred.cpu().numpy()[0].transpose(1, 2, 0)
    except Exception as e:
        logger.error(f"Failed to generate L2A: {e}")
        return None

@benchmark
def generate_plot_band(x_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray, bands: list, cmap: str, output_dir: str) -> None:
    """
    Visualize the results with a simple histogram comparison for prediction vs reference.

    Args:
        x_np: Input data (L1C) array with shape [H, W, C]
        gt_np: Ground truth data (L2A) with shape [H, W, C]
        pred_np: Predicted data (L2A) array with shape [H, W, C]
        bands: List of band names
        cmap: Colormap to use for visualization
        output_dir: Directory to save the output images
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        for idx, band in enumerate(bands):
            # Create a figure with images and a simple histogram
            fig = plt.figure(figsize=(20, 12))

            # Define a grid layout - 2 rows, 4 columns with bigger image row
            grid = plt.GridSpec(2, 4, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

            # Top row - images
            ax_img1 = plt.subplot(grid[0, 0])
            ax_img2 = plt.subplot(grid[0, 1])
            ax_img3 = plt.subplot(grid[0, 2])
            ax_img4 = plt.subplot(grid[0, 3])

            # Bottom row - just one histogram comparing prediction and reference
            ax_hist = plt.subplot(grid[1, :])

            # Get the data for this band
            band_x = x_np[:, :, idx]
            band_gt = gt_np[:, :, idx]
            band_pred = pred_np[:, :, idx]

            # Calculate the difference
            diff_target_pred = (np.abs(band_gt - band_pred) / band_gt) * 100

            # Plot images
            im1 = ax_img1.imshow(band_x, cmap=cmap, vmin=0, vmax=1)
            ax_img1.set_title(f"Input L1C - Band: {band}", fontsize=14)
            ax_img1.axis('off')
            plt.colorbar(im1, ax=ax_img1, fraction=0.046, pad=0.04)

            im2 = ax_img2.imshow(band_gt, cmap=cmap, vmin=0, vmax=1)
            ax_img2.set_title(f"Reference L2A Sen2Cor - Band: {band}", fontsize=14)
            ax_img2.axis('off')
            plt.colorbar(im2, ax=ax_img2, fraction=0.046, pad=0.04)

            im3 = ax_img3.imshow(band_pred, cmap=cmap, vmin=0, vmax=1)
            ax_img3.set_title(f"Prediction L2A - Band: {band}", fontsize=14)
            ax_img3.axis('off')
            plt.colorbar(im3, ax=ax_img3, fraction=0.046, pad=0.04)

            im4 = ax_img4.imshow(diff_target_pred, cmap='plasma', vmin=0, vmax=100)
            ax_img4.set_title(f"Relative Error [%] - {band}", fontsize=14)
            ax_img4.axis('off')
            plt.colorbar(im4, ax=ax_img4, fraction=0.046, pad=0.04)

            # Simple histogram comparison
            # Filter out zeros and NaN values
            gt_data = band_gt[band_gt > 0].flatten()
            pred_data = band_pred[band_pred > 0].flatten()
            # Find common x-axis limits
            min_val = min(gt_data.min(), pred_data.min())
            max_val = max(np.percentile(gt_data, 98), np.percentile(pred_data, 98))

            # Create bins
            bins = np.linspace(min_val, max_val, 100)

            # Plot histograms
            ax_hist.hist(gt_data, bins=bins, alpha=0.5, color='green', label='Reference L2A')
            ax_hist.hist(pred_data, bins=bins, alpha=0.5, color='red', label='Prediction L2A')
            ax_hist.set_title(f"Histogram Comparison - Band {band}", fontsize=14)
            ax_hist.set_xlabel("Pixel Value", fontsize=12)
            ax_hist.set_ylabel("Frequency", fontsize=12)
            ax_hist.legend(fontsize=12)
            ax_hist.set_xlim(0,1)

            # Add metrics
            rmse = np.sqrt(np.mean((band_gt - band_pred)**2))
            mae = np.mean(np.abs(band_gt - band_pred))

            # Add metrics as text to the plot
            ax_hist.text(0.99, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                        transform=ax_hist.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                        fontsize=12)
            # Save the figure
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            fig.savefig(f"{output_dir}/{band}.svg", dpi=300, bbox_inches='tight')
            plt.close(fig)

        logger.success(f"Visualizations with histograms generated in {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())

@benchmark
def generate_tci_plot(x_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray, bands: list, output_dir: str) -> None:
    """
    Generate True Color Image (RGB composite) plots for both input and predicted data.

    Args:
        x_np: Input data array with shape [H, W, C]
        gt_np: Ground true L2A data array with shape [H, W, C]
        pred_np: Predicted data array with shape [H, W, C]
        bands: List of band names
        output_dir: Directory to save the output images
    """
    try:
        # Find indices for RGB bands (B04-Red, B03-Green, B02-Blue)
        rgb_indices = []
        for rgb_band in bands:
            if rgb_band in bands:
                rgb_indices.append(bands.index(rgb_band))
            else:
                logger.error(f"Required band {rgb_band} not found in the available bands")
                return

        if len(rgb_indices) != 3:
            logger.error("Could not find all required RGB bands")
            return
        rgb_indices = rgb_indices[::-1]
        # Extract RGB bands
        rgb_x = x_np[:, :, rgb_indices].copy()  # Make a copy to avoid modifying the original data
        rgb_pred = pred_np[:, :, rgb_indices].copy()
        gt_np = gt_np[:, :, rgb_indices].copy()
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))

        # Plot input TCI
        axs[0].imshow(rgb_x)
        axs[0].set_title(f"Input L1C - True Color Index {bands}", fontsize=16)
        axs[0].axis('off')


        # Plot gt  TCI
        axs[1].imshow(rgb_pred)
        axs[1].set_title(f"Reference L2A Sen2Cor- True Color Index {bands}", fontsize=16)
        axs[1].axis('off')

        # Plot predicted TCI
        axs[2].imshow(rgb_pred)
        axs[2].set_title(f"Predicted L2A - True Color Index {bands}", fontsize=16)
        axs[2].axis('off')

        # Save figure
        fig.tight_layout()
        fig.savefig(f"{output_dir}/TCI.svg", dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.success("TCI RGB composite visualization generated")
    except Exception as e:
        logger.error(f"Failed to generate TCI RGB composite: {e}")



def plot_benchmark_results(function_durations, output_dir):
    # Convert durations from seconds to minutes
    avg_durations = {k: sum(v) / len(v) / 60 for k, v in function_durations.items()}

    # Avoid log scale issues
    epsilon = 1e-6
    avg_durations = {k: max(val, epsilon) for k, val in avg_durations.items()}

    # Hardcoded Sen2Cor processing time in minutes
    sen2cor_time_min = 35

    plt.figure(figsize=(12, 6))
    plt.bar(avg_durations.keys(), avg_durations.values(), color='skyblue')

    plt.yscale('log')
    plt.ylabel('Duration (minutes, log scale)')
    plt.title('Tile Generation Benchmark (Log Scale)')
    plt.xticks(rotation=45, ha='right')

    # Add horizontal line for Sen2Cor processing time
    plt.axhline(y=sen2cor_time_min, color='red', linestyle='--', label='Sen2Cor processing time (35 min)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_results.svg", dpi=300)
    plt.close()


def main() -> None:
    # Set up logging
    logger.add("log_AiSen2Cor.log", rotation="10 MB")
    logger.info("Start workflow ...")
    # Load environment and configs

    env = initialize_env(key_id=sys.argv[1], secret_key=sys.argv[2])
    dir_path = os.getcwd()

    model_cfg = load_config(f"{dir_path}/cfg/config.yaml")
    query_cfg = load_config(f"{dir_path}/cfg/inference_query_config.yaml")
    model_path = f"{dir_path}/weight/AiSen2Cor_EfficientNet_b2.pth"

    # Setup
    endpoint_url = query_cfg["endpoint_url"]
    bucket_name = query_cfg["bucket_name"]
    stac_url = query_cfg["endpoint_stac"]
    s3, s3_client = connect_to_s3(endpoint_url, env["access_key_id"], env["secret_access_key"])
    catalog = pystac_client.Client.open(stac_url)

    # Fetch data
    bands = model_cfg["DATASET"]["bands"]
    bbox = query_cfg["query"]["bbox"]
    start_date = query_cfg["query"]["start_date"]
    end_date = query_cfg["query"]["end_date"]
    max_cloud_cover = query_cfg["query"]["max_cloud_cover"]

    l1c_item, l2a_item  = data_query(catalog, bbox, start_date, end_date, max_cloud_cover)

    l1c_raw_data = load_bands_from_s3(s3_client, bucket_name, l1c_item, bands)
    l2a_raw_data = load_bands_from_s3(s3_client, bucket_name, l2a_item, bands, product_level="L2A")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_cfg["MODEL"], model_path, device)

    # Inference
    resize = model_cfg["TRAINING"]["resize"]
    x_tensor, valid_mask = preprocess(raw_data=l1c_raw_data, resize=resize, device=device)
    gt_tensor, gt_mask = preprocess(raw_data=l2a_raw_data, resize=resize, device=device)
    pred_np = predict(model=model, x_tensor=x_tensor)

    x_np, pred_np = postprocess(x_tensor=x_tensor, pred_tensor=pred_np, valid_mask=valid_mask)
    gt_np, _ = postprocess(x_tensor=gt_tensor, pred_tensor=pred_np, valid_mask=gt_mask)

    # Visualization
    generate_plot_band(x_np=x_np, gt_np=gt_np, pred_np=pred_np, bands=bands, cmap="Grays_r", output_dir=dir_path)
    generate_tci_plot(x_np=x_np, gt_np=gt_np, pred_np=pred_np, bands=bands[::-1], output_dir=dir_path)


    logger.info("Plot tile generation benchmark")

    plot_benchmark_results(function_durations=function_durations, output_dir=dir_path)

    logger.success("Workflow completed")



if __name__ == "__main__":
    main()
