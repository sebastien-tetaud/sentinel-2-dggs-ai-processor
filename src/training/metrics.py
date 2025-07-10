import torch
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import Dict, List, Optional, Tuple

class SpectralAngularMapper(Metric):
    """
    Simplified Spectral Angular Mapper (SAM) metric that uses provided valid masks.
    """

    def __init__(self):
        """Initialize the SAM metric."""
        super().__init__()

        # Add states for tracking
        self.add_state("sum_sam", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> None:
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            valid_mask: Validity mask [B, C, H, W], True for valid pixels
        """
        # If no mask provided, create one assuming all pixels are valid
        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        # Process each sample in batch
        batch_size = preds.shape[0]
        for b in range(batch_size):
            # Get single-channel data for this sample
            pred_b = preds[b, 0]  
            target_b = target[b, 0]
            mask_b = valid_mask[b, 0]

            # Skip if no valid pixels
            if not torch.any(mask_b):
                continue

            # Get valid pixels
            pred_valid = pred_b[mask_b]
            target_valid = target_b[mask_b]

            # Compute dot product and norms
            dot_product = torch.sum(pred_valid * target_valid)
            pred_norm = torch.sqrt(torch.sum(pred_valid ** 2))
            target_norm = torch.sqrt(torch.sum(target_valid ** 2))

            # Compute angle
            cos_sim = dot_product / (pred_norm * target_norm + 1e-8)
            angle = torch.acos(cos_sim)

            # Update state
            self.sum_sam += angle
            self.total_pixels += 1

    def compute(self) -> torch.Tensor:
        """Return the average SAM over all samples."""
        return self.sum_sam / self.total_pixels if self.total_pixels > 0 else torch.tensor(0.0)

# class SpectralAngularMapper(Metric):
#     """
#     Simplified Spectral Angular Mapper (SAM) metric that handles valid pixels.
#     """

#     def __init__(self):
#         """Initialize the SAM metric."""
#         super().__init__()

#         # Add states for tracking
#         self.add_state("sum_sam", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
#         """
#         Update state with predictions and targets.

#         Args:
#             preds: Predictions tensor [B, C, H, W]
#             target: Target tensor [B, C, H, W]
#         """
#         # Create valid mask (True for valid pixels)
#         valid_mask = (target >= 0)

#         # Process each sample in batch
#         batch_size = preds.shape[0]
#         for b in range(batch_size):
#             # Get single-channel data for this sample
#             pred_b = preds[b, 0]  # Assuming single-channel input
#             target_b = target[b, 0]
#             mask_b = valid_mask[b, 0]

#             # Skip if no valid pixels
#             if not torch.any(mask_b):
#                 continue

#             # Get valid pixels
#             pred_valid = pred_b[mask_b]
#             target_valid = target_b[mask_b]

#             # Compute dot product and norms
#             dot_product = torch.sum(pred_valid * target_valid)
#             pred_norm = torch.sqrt(torch.sum(pred_valid ** 2))
#             target_norm = torch.sqrt(torch.sum(target_valid ** 2))

#             # Compute angle
#             cos_sim = dot_product / (pred_norm * target_norm + 1e-8)
#             # cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Prevent numerical errors
#             angle = torch.acos(cos_sim)

#             # Update state
#             self.sum_sam += angle
#             self.total_pixels += 1

#     def compute(self) -> torch.Tensor:
#         """Return the average SAM over all samples."""
#         return self.sum_sam / self.total_pixels if self.total_pixels > 0 else torch.tensor(0.0)

class MultiSpectralMetrics:
    """
    Class for computing and tracking multiple metrics across spectral bands.
    Uses pre-computed valid pixel masks for satellite imagery.
    """

    def __init__(self, bands: List[str], device: str = 'cuda'):
        """
        Initialize metrics for each spectral band.

        Args:
            bands: List of band names
            device: Device to compute metrics on
        """
        self.bands = bands
        self.device = device
        self.metrics = {}

        # Initialize all metrics for all bands
        for band in bands:
            self.metrics[band] = {
                'psnr': PeakSignalNoiseRatio(data_range=1.0).to(device),
                'rmse': MeanSquaredError(squared=False).to(device),
                'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
                'sam': SpectralAngularMapper().to(device),
            }

    def update(self, outputs: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None) -> None:
        """
        Update metrics for all bands.

        Args:
            outputs: Model predictions [B, C, H, W]
            targets: Ground truth [B, C, H, W]
            valid_mask: Pre-computed validity mask [B, C, H, W], True for valid pixels
        """
        # If no mask is provided, create a default one (all True)
        if valid_mask is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        
        for c, band in enumerate(self.bands):
            # Extract channel data
            outputs_c = outputs[:, c:c+1, :, :]  # Keep dim for SSIM
            targets_c = targets[:, c:c+1, :, :]
            mask_c = valid_mask[:, c:c+1, :, :]  # Get mask for this channel

            # Only process if we have valid pixels
            if torch.any(mask_c):
                # For metrics that need 2D input (dropping channel dim)
                outputs_valid = outputs_c.squeeze(1)[mask_c.squeeze(1)]
                targets_valid = targets_c.squeeze(1)[mask_c.squeeze(1)]

                self.metrics[band]['psnr'].update(outputs_valid, targets_valid)
                self.metrics[band]['rmse'].update(outputs_valid, targets_valid)

                # For metrics that need 4D input (keeping batch and channel dims)
                if mask_c.all():
                    # If all pixels are valid, use tensors as is
                    self.metrics[band]['ssim'].update(outputs_c, targets_c)
                else:
                    # Handle partial valid pixels for SSIM which needs full images
                    # Apply the mask to create copies with zeros in invalid areas
                    masked_outputs = outputs_c * mask_c.float()
                    masked_targets = targets_c * mask_c.float()
                    
                    self.metrics[band]['ssim'].update(masked_outputs, masked_targets)

                # Update SAM with both data and mask
                self.metrics[band]['sam'].update(outputs_c, targets_c, mask_c)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics for all bands.

        Returns:
            Dictionary with metrics for each band
        """
        results = {}
        for band in self.bands:
            results[band] = {}
            for metric_name, metric in self.metrics[band].items():
                results[band][metric_name] = metric.compute().item()
        return results

    def reset(self) -> None:
        """Reset all metrics."""
        for band in self.bands:
            for metric in self.metrics[band].values():
                metric.reset()

# class MultiSpectralMetrics:
#     """
#     Class for computing and tracking multiple metrics across spectral bands.
#     Handles valid pixel masking for satellite imagery.
#     """

#     def __init__(self, bands: List[str], device: str = 'cuda'):
#         """
#         Initialize metrics for each spectral band.

#         Args:
#             bands: List of band names
#             device: Device to compute metrics on
#         """
#         self.bands = bands
#         self.device = device
#         self.metrics = {}

#         # Initialize all metrics for all bands
#         for band in bands:
#             self.metrics[band] = {
#                 'psnr': PeakSignalNoiseRatio(data_range=1.0).to(device),
#                 'rmse': MeanSquaredError(squared=False).to(device),
#                 'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
#                 'sam': SpectralAngularMapper().to(device),
#             }

#     def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
#         """
#         Update metrics for all bands.

#         Args:
#             outputs: Model predictions [B, C, H, W]
#             targets: Ground truth [B, C, H, W]
#         """
#         for c, band in enumerate(self.bands):
#             # Extract channel data
#             outputs_c = outputs[:, c:c+1, :, :]  # Keep dim for SSIM
#             targets_c = targets[:, c:c+1, :, :]

#             # Create channel-wise valid mask (True for valid pixels)
#             valid_mask_c = (targets_c >= 0)

#             # Only process if we have valid pixels
#             if torch.any(valid_mask_c):
#                 # For metrics that need 2D input (dropping channel dim)
#                 outputs_valid = outputs_c.squeeze(1)[valid_mask_c.squeeze(1)]
#                 targets_valid = targets_c.squeeze(1)[valid_mask_c.squeeze(1)]

#                 self.metrics[band]['psnr'].update(outputs_valid, targets_valid)
#                 self.metrics[band]['rmse'].update(outputs_valid, targets_valid)

#                 # For metrics that need 4D input (keeping batch and channel dims)
#                 if valid_mask_c.all():
#                     # If all pixels are valid, use tensors as is
#                     self.metrics[band]['ssim'].update(outputs_c, targets_c)
#                 else:
#                     # Handle partial valid pixels for SSIM which needs full images
#                     masked_outputs = outputs_c.clone()
#                     masked_targets = targets_c.clone()
#                     masked_outputs[~valid_mask_c] = 0
#                     masked_targets[~valid_mask_c] = 0

#                     self.metrics[band]['ssim'].update(masked_outputs, masked_targets)

#                 # For SAM, pass the tensors directly - it handles masking internally
#                 self.metrics[band]['sam'].update(outputs_c, targets_c)

#     def compute(self) -> Dict[str, Dict[str, float]]:
#         """
#         Compute all metrics for all bands.

#         Returns:
#             Dictionary with metrics for each band
#         """
#         results = {}
#         for band in self.bands:
#             results[band] = {}
#             for metric_name, metric in self.metrics[band].items():
#                 results[band][metric_name] = metric.compute().item()
#         return results

#     def reset(self) -> None:
#         """Reset all metrics."""
#         for band in self.bands:
#             for metric in self.metrics[band].values():
#                 metric.reset()


def avg_metric_bands(val_metrics, metric_name):
    """
    Compute the average of a given metric_name across all bands.

    Parameters:
    -----------
    val_metrics : dict
        Dictionary with band names as keys, each containing a dictionary
        with metrics including 'sam'.
    metric_name: str
        metric name: e.g: sam
    Returns:
    --------
    float
        The average SAM value across all bands.
    """
    total_sam = 0.0
    band_count = len(val_metrics.keys())

    for band, metrics in val_metrics.items():
        total_sam += metrics[metric_name]

    return total_sam / band_count


