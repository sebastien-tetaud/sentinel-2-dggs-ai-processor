import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(
    df,
    bands=['B02', 'B03', 'B04'],
    log_scale=False,
    title="SSIM Metrics Over Training Epochs",
    y_label="SSIM",
    verbose=False,
    save=False,
    save_path="./",
    color_palette="plasma",
):
    """
    Plot training and validation metrics for multiple spectral bands.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing metrics data with 'epoch' column and metrics columns
        formatted as 'train_{band}' and 'val_{band}'.
    bands : list, optional
        List of band names to plot, default is ['B02', 'B03', 'B04'].
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis, default is False.
    title : str, optional
        Plot title, default is "SSIM Metrics Over Training Epochs".
    y_label : str, optional
        Y-axis label, default is "SSIM".
    verbose : bool, optional
        Whether to display the plot, default is False.
    save : bool, optional
        Whether to save the plot, default is False.
    save_path : str, optional
        Directory path to save the figure, default is "./".
    color_palette : str, optional
        Name of seaborn color palette to use, default is "plasma".

    Returns
    -------
    None
        Function creates and optionally saves/displays a plot.
    """
    # Set up color palette
    colors = sns.color_palette(color_palette, len(bands))

    # Create figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Loop through bands to plot
    for i, band in enumerate(bands):
        # Plot training curves (dashed)
        ax.plot(
            df['epoch'],
            df[f'train_{band}'],
            '--',
            label=f'Train {band}',
            color=colors[i],
        )

        # Plot validation curves (solid)
        ax.plot(
            df['epoch'],
            df[f'val_{band}'],
            label=f'Val {band}',
            color=colors[i],
        )

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Set tight layout for better spacing
    plt.tight_layout()

    # Save plot if requested
    if save:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Construct filename based on metric type
        filename = f"{y_label.lower().replace(' ', '_')}_metric_training.svg"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')

        if verbose:
            print(f"Plot saved to: {full_path}")

    # Display plot if requested
    if verbose:
        plt.show()

    plt.close()


def plot_training_loss(
    df,
    title="Training and Validation Loss",
    y_label="Loss",
    log_scale=False,
    verbose=False,
    save=False,
    save_path="./",
    color_palette="RdBu_r",
):
    """
    Plot training and validation loss over epochs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'epoch', 'train_loss', and 'val_loss' columns.
    title : str, optional
        Plot title, default is "Training and Validation Loss".
    y_label : str, optional
        Y-axis label, default is "Loss".
    log_scale : bool, optional
        Whether to use logarithmic scale for y-axis, default is False.
    verbose : bool, optional
        Whether to display the plot, default is False.
    save : bool, optional
        Whether to save the plot, default is False.
    save_path : str, optional
        Directory path to save the figure, default is "./".
    color_palette : str, optional
        Name of seaborn color palette to use, default is "plasma".

    Returns
    -------
    None
        Function creates and optionally saves/displays a plot.
    """
    # Set up color palette
    colors = sns.color_palette(color_palette, 2)

    # Create figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.plot(df["epoch"], df["train_loss"], label='Training Loss', color=colors[0])
    ax.plot(df["epoch"], df["val_loss"], label='Validation Loss', color=colors[1])

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Set tight layout for better spacing
    plt.tight_layout()

    # Save plot if requested
    if save:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Construct filename
        filename = "loss_plot.svg"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')

        if verbose:
            print(f"Plot saved to: {full_path}")

    # Display plot if requested
    if verbose:
        plt.show()

    plt.close()
