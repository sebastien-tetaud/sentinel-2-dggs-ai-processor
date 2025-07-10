import torch
import torch.nn as nn
import math

def masked_mse_loss(predictions, targets, valid_mask):
    """
    Compute MSE loss only on valid pixels.

    Args:
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth images (B, C, H, W)
        valid_mask: Binary mask indicating valid pixels (B, 1, H, W) or (B, C, H, W)

    Returns:
        Masked MSE loss computed only on valid pixels
    """
    # Apply mask
    masked_pred = predictions * valid_mask
    masked_target = targets * valid_mask

    # Count valid pixels for normalization
    num_valid_pixels = torch.sum(valid_mask) + 1e-8

    # Compute MSE loss on valid pixels only
    loss = torch.sum((masked_pred - masked_target) ** 2) / num_valid_pixels

    return loss




class WeightedMSELoss(nn.Module):
    """
    MSE loss weighted by valid pixel percentages.
    Only considers pixels where target > 0 as valid.
    Weights each channel's contribution by sqrt(valid_pixel_percent)/100.
    """

    def __init__(self, reduction='mean'):
        """
        Initialize the weighted MSE loss.

        Args:
            reduction: Specifies the reduction applied to the output (for compatibility with nn.MSELoss)
                       'none': no reduction, 'mean': mean of losses, 'sum': sum of losses
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate MSE loss weighted by valid pixel percentages.

        Args:
            outputs: Model predictions [B, C, H, W]
            targets: Ground truth [B, C, H, W]

        Returns:
            Weighted loss value
        """
        batch_size, num_channels = outputs.shape[0], outputs.shape[1]
        total_loss = 0.0
        total_weight = 0.0

        for b in range(batch_size):
            for c in range(num_channels):
                # Extract single channel
                output_bc = outputs[b, c]
                target_bc = targets[b, c]

                # Create valid mask
                valid_mask = (target_bc > 0)

                # Skip if no valid pixels
                if not torch.any(valid_mask):
                    continue

                # Calculate valid pixel percentage
                total_pixels = valid_mask.numel()
                valid_pixels = torch.sum(valid_mask).item()
                valid_pixel_percent = (valid_pixels / total_pixels)

                # Calculate weight: sqrt(valid_pixel_percent)/100
                weight = math.sqrt(valid_pixel_percent)

                # Calculate MSE only on valid pixels
                if valid_pixels > 0:
                    # Extract valid pixels
                    output_valid = output_bc[valid_mask]
                    target_valid = target_bc[valid_mask]

                    # Calculate MSE
                    mse = torch.mean((output_valid - target_valid) ** 2)

                    # Add weighted loss
                    total_loss += mse * weight
                    total_weight += weight

        # Return loss based on reduction method
        if total_weight > 0:
            if self.reduction == 'none':
                return total_loss
            elif self.reduction == 'sum':
                return total_loss
            else:  # 'mean' is default
                return total_loss / total_weight
        else:
            return torch.tensor(0.0, device=outputs.device)

