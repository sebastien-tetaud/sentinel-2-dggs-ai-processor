import os
import sys
from tqdm import tqdm
import natsort
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import cv2
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from PIL import Image


# Set project-related paths
src_dir = os.path.abspath('')
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

from data.loader import define_loaders
from utils.utils import load_config, prepare_paths
from utils.torch import load_model_weights
from model_zoo.models import define_model
from utils.plot import plot_metrics, plot_training_loss
from data.dataset import Sentinel2Dataset, read_images, normalize
from training.metrics import MultiSpectralMetrics

def generate_exp_paths(exp_path):
    """
    Generate paths for results, checkpoints, metrics, and logs.

    Returns:
        dict: A dictionary containing various paths related to results.
    """

    return {

        "result_dir": exp_path,
        "checkpoint_path": os.path.join(exp_path, "checkpoints"),
        "metrics_path": os.path.join(exp_path, "metrics"),
        "log_path": os.path.join(exp_path, "training.log")
    }


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


def prepare_data(config):
    """
    Prepare the data loader for test datasets.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        DataLoader: Data loader for the test dataset.
    """
    version = config['DATASET']['version']
    resize = config['TRAINING']['resize']

    TEST_DIR = f"/mnt/disk/dataset/sentinel-ai-processor/{version}/test/"
    df_test_input, df_test_output = prepare_paths(TEST_DIR)

    test_dataset = Sentinel2Dataset(df_x=df_test_input, df_y=df_test_output,
                                     train=True, augmentation=False, img_size=resize)
    logger.info(df_test_output.head(5))

    return define_loaders(
        train_dataset=test_dataset,
        val_dataset=None,
        train=False,
        batch_size=config['TRAINING']['batch_size'],
        num_workers=config['TRAINING']['num_workers']
    )


def evaluate_and_plot(model, df_test_input, df_test_output, bands,cmap,  resize, device, index, verbose, save, output_path):
    """
    Evaluates and plots input, target, prediction, and absolute difference for a specific index.

    Parameters:
        model: Trained model for evaluation.
        df_test_input: DataFrame containing the input data paths.
        df_test_output: DataFrame containing the output data paths.
        bands: List of band names (e.g., ['B02', 'B03', 'B04']) to evaluate.
        resize: The resize dimension for images.
        device: The device (CPU or GPU) to run the model on.
        index: The specific index to evaluate from the data.
    """
    # Load input data and mask
    x_paths = natsort.natsorted(glob.glob(os.path.join(df_test_input["path"][index], "*.png"), recursive=False))
    x_data = read_images(x_paths)
    x_data, x_mask = normalize(x_data)
    x_data = cv2.resize(x_data, (resize, resize), interpolation=cv2.INTER_AREA)
    x_mask = cv2.resize(x_mask.astype(np.uint8), (resize, resize), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Load output data and mask
    y_paths = natsort.natsorted(glob.glob(os.path.join(df_test_output["path"][index], "*.png"), recursive=False))
    y_data = read_images(y_paths)
    y_data, y_mask  = normalize(y_data)
    y_data = cv2.resize(y_data, (resize, resize), interpolation=cv2.INTER_AREA)
    y_mask = cv2.resize(y_mask.astype(np.uint8), (resize, resize), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Combine masks: valid only where both input and target are valid
    valid_mask = x_mask

    # Prepare tensors for inference
    x_tensor = torch.from_numpy(x_data).float().permute(2, 0, 1).unsqueeze(0).to(device)  # [1, C, H, W]

    # Inference
    model.eval()
    with torch.no_grad():
        pred_tensor = model(x_tensor)

    # Convert tensors to NumPy
    x_np = x_tensor.cpu().numpy()[0].transpose(1, 2, 0)       # [H, W, C]
    y_np = torch.from_numpy(y_data).numpy()                   # [H, W, C]
    pred_np = pred_tensor.cpu().numpy()[0].transpose(1, 2, 0) # [H, W, C]

    # Apply mask: set invalid pixels to 0
    # x_np[~valid_mask] = 0.0
    y_np[~valid_mask] = 0.0
    pred_np[~valid_mask] = 0.0

    # Plot results for each band
    for idx, band in enumerate(bands):
        fig, axs = plt.subplots(1, 4, figsize=(20, 6))

        vmin = 0
        vmax = 1  # Normalized range

        # Input
        im0 = axs[0].imshow(x_np[:, :, idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title(f"L1C Input - {band}", fontsize=14)
        axs[0].axis('off')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        # Target
        im1 = axs[1].imshow(y_np[:, :, idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title(f"L2A Target - {band}", fontsize=14)
        axs[1].axis('off')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # Prediction
        im2 = axs[2].imshow(pred_np[:, :, idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[2].set_title(f" L2A Prediction - {band}", fontsize=14)
        axs[2].axis('off')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        # Absolute Difference
        diff_target_pred = np.abs(y_np[:, :, idx] - pred_np[:, :, idx])
        im3 = axs[3].imshow(diff_target_pred, cmap=cmap, vmin=0, vmax=diff_target_pred.max())
        axs[3].set_title(f"Abs Difference - {band}", fontsize=14)
        axs[3].axis('off')
        plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if verbose:
            plt.show()

        if save:
            _, head = os.path.split(df_test_input["path"][index])
            filename = f"{output_path}/{head}_{band}.svg"
            fig.savefig(filename)

        plt.close()


def calculate_valid_pixel_percentages(df, column_name="path", show_progress=True):
    """
    Calculate the percentage of valid (non-zero) pixels for each entry in the dataframe.

    Args:
        df (DataFrame): DataFrame containing paths to image folders.
        column_name (str): Name of the column in the dataframe that contains the image folder paths.
        show_progress (bool): Whether to show a progress bar.

    Returns:
        DataFrame: DataFrame with an additional column for valid pixel percentages.
    """
    valid_pixel_percentages = []

    # Create iterator with or without progress bar
    iterator = tqdm(range(len(df)), desc="Processing Valid Pixel", ncols=100, colour='#ff6666') if show_progress else range(len(df))

    for i in iterator:
        x_paths = natsort.natsorted(glob.glob(os.path.join(df[column_name][i], "*.png"), recursive=False))

        if not x_paths:
            valid_pixel_percentages.append(0)
            continue

        data = Image.open(x_paths[0])
        data = np.array(data)

        total_pixels = data.size
        pixels_greater_than_zero = np.sum(data > 0)
        percentage = (pixels_greater_than_zero / total_pixels) * 100
        valid_pixel_percentages.append(percentage)

    df['valid_pixel'] = valid_pixel_percentages
    return df


def plot_3d_scatter(
    df,
    x_col,
    y_col,
    z_col,
    color_col=None,
    labels=None,
    title=None,
    output_path=None,
    color_scale='plasma',
    opacity=0.8,
    marker_size=5
):
    if not labels:
        labels = {x_col: x_col, y_col: y_col, z_col: z_col}
    if not title:
        title = f'{z_col} vs {x_col} vs {y_col}'
    if not color_col:
        color_col = x_col  # default to x_col if not specified

    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        color_continuous_scale=color_scale,
        opacity=opacity,
        title=title,
        labels=labels
    )

    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(
        scene=dict(
            xaxis_title=labels.get(x_col, x_col),
            yaxis_title=labels.get(y_col, y_col),
            zaxis_title=labels.get(z_col, z_col),
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            zaxis=dict(gridcolor='lightgray'),
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=30),
        coloraxis_colorbar=dict(title=labels.get(color_col, color_col))
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def post_traing_analysis(path):

    # Sim path during the training
    exp_paths = generate_exp_paths(exp_path=path)
    config = load_config(config_path=f"{exp_paths['result_dir']}/config.yaml")

    base_dir = config["DATASET"]["base_dir"]
    version = config['DATASET']['version']
    bands = config['DATASET']['bands']
    resize = config['TRAINING']['resize']
    num_workers = config['TRAINING']['num_workers']

    df_loss = pd.read_csv(f"{exp_paths['metrics_path']}/losses.csv")
    df_ssim = pd.read_csv(f"{exp_paths['metrics_path']}/ssim_metrics.csv")
    df_sam = pd.read_csv(f"{exp_paths['metrics_path']}/sam_metrics.csv")
    df_rmse = pd.read_csv(f"{exp_paths['metrics_path']}/rmse_metrics.csv")
    df_psnr = pd.read_csv(f"{exp_paths['metrics_path']}/psnr_metrics.csv")

    # Plot all metrics
    metrics = {
        "PSNR": df_psnr,
        "SSIM": df_ssim,
        "SAM": df_sam,
        "RMSE": df_rmse
    }
    for title, df_metric in metrics.items():
        plot_metrics(df_metric, bands=bands, title=f"{title} Evolution During Training - Dataset {version}",
                    log_scale=False, y_label=title, verbose=False, save=True,
                    save_path=exp_paths['metrics_path'], color_palette="plasma")

    plot_training_loss(df=df_loss,
                    title="Training and Validation Loss",
                    y_label="Loss",
                    log_scale=False,
                    verbose=False,
                    save=True,
                    save_path=exp_paths['metrics_path'],
                    color_palette="plasma"
                    )
    # Load test data
    test_dir = f"/mnt/disk/dataset/sentinel-ai-processor/{version}/test/"
    df_test_input, df_test_output = prepare_paths(test_dir)

    df_test_output = calculate_valid_pixel_percentages(df=df_test_output, column_name="path", show_progress=True)

    test_dataset = Sentinel2Dataset(df_x=df_test_input, df_y=df_test_output, train=True, augmentation=False, img_size=resize)
    test_loader = define_loaders(
        train_dataset=test_dataset,
        val_dataset=None,
        train=False,
        batch_size=1,
        num_workers=num_workers
    )

    weights_path = f"{exp_paths['checkpoint_path']}/best_model.pth"
    model = define_model(name=config["MODEL"]["model_name"],
                        encoder_name=config["MODEL"]["encoder_name"],
                        in_channel=len(bands),
                        out_channels=len(bands),
                        activation=config["MODEL"]["activation"])

    # Load best model weights
    model = load_model_weights(model=model, filename=weights_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_metrics_tracker = MultiSpectralMetrics(bands=bands, device=device)
    metrics_dict = {f"{metric}_{band}": [] for band in test_metrics_tracker.bands for metric in ['psnr', 'rmse', 'ssim', 'sam']}

    # Reset tracker and model evaluation
    model.eval()
    test_metrics_tracker.reset()

    with torch.no_grad():
        with tqdm(total=len(test_loader.dataset), ncols=100, colour='#cc99ff') as t:
            for x_data, y_data, valid_mask in test_loader:
                x_data, y_data, valid_mask = x_data.to(device), y_data.to(device), valid_mask.to(device)
                outputs = model(x_data)
                test_metrics_tracker.reset()
                test_metrics_tracker.update(outputs, y_data, valid_mask)
                metrics = test_metrics_tracker.compute()

                for band in test_metrics_tracker.bands:
                    for metric_name in ['psnr', 'rmse', 'ssim', 'sam']:
                        metrics_dict[f"{metric_name}_{band}"].append(metrics[band][metric_name])

                t.update(x_data.size(0))

    # Update DataFrame with metrics
    for column_name, values in metrics_dict.items():
        df_test_output[column_name] = values

    # Visualization for SAM vs Cloud Cover
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, len(bands), figsize=(18, 6))
    cmap = plt.cm.plasma
    global_ymax = max(df_test_output[f'sam_{band}'].max() for band in bands) * 1.05  # 5% buffer

    for i, band in enumerate(bands):
        ax = axs[i]
        scatter = ax.scatter(data=df_test_output, x='cloud_cover', y=f'sam_{band}',
                            c=df_test_output['cloud_cover'], cmap=cmap)
        ax.set_title(f"{band} - Test Data", fontsize=14)
        ax.set_xlabel("Cloud Cover (%)", fontsize=12)
        ax.set_ylabel("SAM", fontsize=12)
        ax.set_ylim(0, global_ymax)
        plt.colorbar(scatter, ax=ax, label="Cloud Cover (%)")

    plt.tight_layout()
    plt.savefig(f"{exp_paths['metrics_path']}/sam_vs_cloud_cover.svg")
    plt.close()

    # Visualization for SSIM vs Cloud Cover
    fig, axs = plt.subplots(1, len(bands), figsize=(18, 6))
    global_ymin = min(df_test_output[f'ssim_{band}'].min() for band in bands) * 0.98  # 5% buffer

    for i, band in enumerate(bands):
        ax = axs[i]
        scatter = ax.scatter(data=df_test_output, x='cloud_cover', y=f'ssim_{band}',
                            c=df_test_output['cloud_cover'], cmap=cmap)
        ax.set_title(f"{band} - Test Data", fontsize=14)
        ax.set_xlabel("Cloud Cover (%)", fontsize=12)
        ax.set_ylabel("SSIM", fontsize=12)
        ax.set_ylim(global_ymin, 1.0)
        plt.colorbar(scatter, ax=ax, label="Cloud Cover (%)")

    plt.tight_layout()
    plt.savefig(f"{exp_paths['metrics_path']}/ssim_vs_cloud_cover.svg")
    plt.close()

    # Visualization for SAM vs Valid Pixel
    fig, axs = plt.subplots(1, len(bands), figsize=(18, 6))
    global_ymax = max(df_test_output[f'sam_{band}'].max() for band in bands) * 1.05

    for i, band in enumerate(bands):
        ax = axs[i]
        scatter = ax.scatter(data=df_test_output, x='valid_pixel', y=f'sam_{band}',
                            c=df_test_output['valid_pixel'], cmap=cmap)
        ax.set_title(f"{band} - Test Data", fontsize=14)
        ax.set_xlabel("Valid Pixel (%)", fontsize=12)
        ax.set_ylabel(f"SAM {band}", fontsize=12)
        ax.set_ylim(0, global_ymax)
        plt.colorbar(scatter, ax=ax, label="Valid Pixel (%)")

    plt.tight_layout()
    plt.savefig(f"{exp_paths['metrics_path']}/sam_vs_valid_pixel.svg")
    plt.close()

    # Create a 3D scatter plot for SAM B03 vs Cloud Cover vs Valid Pixels
    import plotly.express as px

    plot_3d_scatter(
        df=df_test_output,
        x_col='cloud_cover',
        y_col='valid_pixel',
        z_col='sam_B02',
        color_col='cloud_cover',
        labels={
            'cloud_cover': 'Cloud Cover (%)',
            'valid_pixel': 'Valid Pixels (%)',
            'sam_B02': 'SAM B02'
        },
        title='3D Scatter: SAM B02 vs Cloud Cover vs Valid Pixels',
        output_path=f"{exp_paths['metrics_path']}/sam_vs_valid_pixel_cloud_cover.html"
    )



    # Top worst predictions for SAM and SSIM
    top_10_min_ssim_idx = df_test_output.sort_values(f'ssim_B02').head(10).index.tolist()
    top_10_max_sam_idx = df_test_output.sort_values(f'sam_B02', ascending=False).head(10).index.tolist()

    logger.info(f"Top 10 indices with minimum SSIM for B02: {top_10_min_ssim_idx}")
    logger.info(f"Top 10 indices with maximum SAM for B02: {top_10_max_sam_idx}")

    # Evaluate and plot for worst SAM predictions
    outputs_worst_sam_path = f"{exp_paths['metrics_path']}/outputs_worst_sam"
    os.makedirs(outputs_worst_sam_path, exist_ok=True)

    for idx in top_10_max_sam_idx:
        evaluate_and_plot(model, df_test_input, df_test_output, bands=bands,cmap="Greys_r", resize=resize,
                        device=device, index=idx, verbose=False, save=True, output_path=outputs_worst_sam_path)

    # Evaluate and plot for worst SSIM predictions
    outputs_worst_ssim_path = f"{exp_paths['metrics_path']}/outputs_worst_ssim"
    os.makedirs(outputs_worst_ssim_path, exist_ok=True)

    for idx in top_10_min_ssim_idx:
        evaluate_and_plot(model, df_test_input, df_test_output, bands=bands,cmap="Greys_r", resize=resize,
                        device=device, index=idx, verbose=False, save=True, output_path=outputs_worst_ssim_path)

    # Top best predictions for SAM and SSIM
    top_10_max_ssim_idx = df_test_output.sort_values(f'ssim_B02').tail(20).index.tolist()
    top_10_min_sam_idx = df_test_output.sort_values(f'sam_B02', ascending=False).tail(20).index.tolist()

    logger.info(f"Top 10 indices with maximum SSIM for B02: {top_10_max_ssim_idx}")
    logger.info(f"Top 10 indices with minimum SAM for B02: {top_10_min_sam_idx}")

    # Evaluate and plot for best SAM predictions
    output_best_sam_path = f"{exp_paths['metrics_path']}/outputs_best_sam"
    os.makedirs(output_best_sam_path, exist_ok=True)

    for idx in top_10_min_sam_idx:
        evaluate_and_plot(model, df_test_input, df_test_output, bands=bands,cmap="Greys_r", resize=resize,
                        device=device, index=idx, verbose=False, save=True, output_path=output_best_sam_path)

    # Evaluate and plot for best SSIM predictions
    output_best_ssim_path = f"{exp_paths['metrics_path']}/outputs_best_ssim"
    os.makedirs(output_best_ssim_path, exist_ok=True)

    for idx in top_10_max_ssim_idx:
        evaluate_and_plot(model, df_test_input, df_test_output, bands=bands,cmap="Greys_r", resize=resize,
                        device=device, index=idx, verbose=False, save=True, output_path=output_best_ssim_path)


# post_traing_analysis(path="/home/ubuntu/project/sentinel-2-ai-processor/src/results/2025-06-12_05-17-04")