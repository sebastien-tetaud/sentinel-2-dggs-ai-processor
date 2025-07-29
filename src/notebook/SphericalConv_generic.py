import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import healpy as hp
import gc

class NeighborIndexProcessor:
    """External processor for building neighbor indices - OUTSIDE the model"""

    def __init__(self, level, nest=True):
        self.level = level
        self.NSIDE = 2 ** level
        self.nest = nest

        # Cache for computed indices
        self.cache = {}

    def build_neighbor_indices(self, available_cell_ids, stride=1):
        """
        Build neighbor indices for given cell IDs and stride

        Args:
            available_cell_ids: Array of HEALPix cell IDs for this chunk
            stride: Stride for center cell sampling

        Returns:
            torch.Tensor: Data neighbor indices [N_patches, 9]
        """
        # Create cache key
        cache_key = (tuple(sorted(available_cell_ids)), stride)

        if cache_key in self.cache:
            return self.cache[cache_key]

        available_cell_set = set(available_cell_ids)
        neighbor_indices = []

        # Create cell_id to data_index mapping
        cell_to_data_idx = {cell_id: i for i, cell_id in enumerate(available_cell_ids)}

        # Apply stride to center cell list
        center_cells = available_cell_ids[::stride]

        for cell_id in center_cells:
            neighbors = hp.get_all_neighbours(self.NSIDE, cell_id, nest=self.nest)

            # Validate each neighbor; replace invalid or missing with center
            valid_neighbors = [
                n if (n != -1 and n in available_cell_set) else cell_id
                for n in neighbors
            ]

            patch = [cell_id] + valid_neighbors  # Center + 8 neighbors

            # Convert to data indices
            patch_data_indices = [cell_to_data_idx[cell_id] for cell_id in patch]
            neighbor_indices.append(patch_data_indices)

        # Convert to tensor
        data_neighbor_indices = torch.tensor(neighbor_indices, dtype=torch.long)

        # Cache the result
        self.cache[cache_key] = data_neighbor_indices

        return data_neighbor_indices

    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()


class SphericalConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        """Spherical Conv without fixed cell IDs"""
        super(SphericalConv, self).__init__()

        # Only the convolution layer - no neighbor processing
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9, bias=bias)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x, neighbor_indices):
        """
        Forward pass with external neighbor indices

        Args:
            x: Input tensor [B, C_in, N_cells]
            neighbor_indices: Pre-computed neighbor indices [N_patches, 9]
        """
        batch_size, n_channels, n_cells = x.shape

        # Move neighbor_indices to same device as input
        if neighbor_indices.device != x.device:
            neighbor_indices = neighbor_indices.to(x.device)

        # Extract patches using the neighbor indices
        # Shape: [B, C_in, N_patches, 9]
        patches = x[:, :, neighbor_indices]

        # Reshape to [B, C_in, N_patches * 9] for Conv1d
        patches_flat = patches.view(batch_size, n_channels, -1)

        # Apply convolution
        output = self.conv(patches_flat)

        return output


class SphericalConvBlock(nn.Module):
    """Spherical conv block without fixed cell IDs"""

    def __init__(self, in_channels, out_channels):
        super(SphericalConvBlock, self).__init__()

        self.conv = SphericalConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, neighbor_indices):
        x = self.conv(x, neighbor_indices)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SphericalDoubleConvBlock(nn.Module):
    """Double spherical conv block without fixed cell IDs"""

    def __init__(self, in_channels, out_channels):
        super(SphericalDoubleConvBlock, self).__init__()

        self.conv1 = SphericalConvBlock(in_channels, out_channels)
        self.conv2 = SphericalConvBlock(out_channels, out_channels)

    def forward(self, x, neighbor_indices):
        x = self.conv1(x, neighbor_indices)
        x = self.conv2(x, neighbor_indices)
        return x


class Model(nn.Module):
    """Model without fixed cell IDs - completely flexible"""

    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()

        self.double_conv = SphericalDoubleConvBlock(in_channels, out_channels)

    def forward(self, x, neighbor_indices):
        """
        Forward pass with external neighbor indices

        Args:
            x: Input tensor [B, C_in, N_cells]
            neighbor_indices: Pre-computed neighbor indices [N_patches, 9]
        """
        return self.double_conv(x, neighbor_indices)

# Memory management function
def clear_memory():
    """Clear GPU and CPU memory"""
    vars_to_delete = ['model', 'x_tensor', 'output', 'x_multi_band', 'spectral_data', 'neighbor_indices']
    for var in vars_to_delete:
        if var in globals():
            del globals()[var]

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("🧹 Memory cleared")


def main():
    """Your original workflow adapted to separated approach"""

    # Clear memory before starting
    clear_memory()

    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load your data
    ds_healpix = xr.open_dataset("/home/ubuntu/project/sentinel-2-dggs-ai-processor/src/notebook/healpix.zarr")
    available_cell_ids = ds_healpix.cell_ids.values
    print(f"Number of available HEALPix cells: {len(available_cell_ids)}")

    # Parameters
    level = 19
    band_list = ['b02', 'b03', 'b04', 'b08']
    in_channels = len(band_list)
    stride = 1

    # 1. Create neighbor processor and compute indices
    neighbor_processor = NeighborIndexProcessor(level=level)
    neighbor_indices = neighbor_processor.build_neighbor_indices(available_cell_ids, stride)
    print(f"Generated {neighbor_indices.shape[0]} patches")

    # 2. Create model (no cell IDs!)
    model = Model(in_channels=in_channels, out_channels=in_channels)
    model = model.to(device)

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 3. Load spectral data
    spectral_data = []
    for band in band_list:
        band_data = ds_healpix.Sentinel2.sel(bands=band).compute().values
        spectral_data.append(band_data)

    x_multi_band = np.stack(spectral_data, axis=0)
    print(f"Multi-band data shape: {x_multi_band.shape}")

    x_tensor = torch.tensor(x_multi_band, dtype=torch.float32).unsqueeze(0)
    print(f"Input tensor shape: {x_tensor.shape}")

    x_tensor = x_tensor.to(device)

    # 4. Forward pass with neighbor indices
    with torch.no_grad():
        output = model(x_tensor, neighbor_indices)
        print(f"Output tensor shape: {output.shape}")

    # Clear memory after execution
    clear_memory()

    return model, output, neighbor_processor

if __name__ == "__main__":

    main()