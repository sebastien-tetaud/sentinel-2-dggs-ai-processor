import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import healpy as hp

class RegionalSphericalConv(nn.Module):
    def __init__(self, available_cell_ids, level, in_channels, out_channels, bias=True, nest=True, stride=1):
        """Regional Spherical Convolutional layer for HEALPix data covering a specific area."""
        super(RegionalSphericalConv, self).__init__()

        self.level = level
        self.NSIDE = 2 ** level
        self.nest = nest
        self.stride = stride
        self.available_cell_ids = np.array(available_cell_ids)
        self.available_cell_set = set(available_cell_ids)

        # Build neighbor index using your strategy
        self.neighbor_indices = self._build_neighbor_index()
        self.n_patches = self.neighbor_indices.shape[0]

        # Create cell_id to data_index mapping
        self.cell_to_data_idx = {cell_id: i for i, cell_id in enumerate(self.available_cell_ids)}

        # Convert neighbor indices to data indices for efficient lookup
        data_neighbor_indices = self._convert_to_data_indices()

        # Register as buffer so it moves to GPU with the model
        self.register_buffer('data_neighbor_indices', data_neighbor_indices)

        # 1D convolution with kernel size 9 (3x3 patch flattened)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9, bias=bias)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)

    def _build_neighbor_index(self):
        """Build 9-cell neighborhood index"""
        available_cell_ids = set(self.available_cell_ids)
        neighbor_indices = []

        # Apply stride to center cell list
        center_cells = self.available_cell_ids[::self.stride]

        for cell_id in center_cells:
            neighbors = hp.get_all_neighbours(self.NSIDE, cell_id, nest=self.nest)

            # Validate each neighbor; replace invalid or missing with center
            valid_neighbors = [
                n if (n != -1 and n in available_cell_ids) else cell_id
                for n in neighbors
            ]

            patch = [cell_id] + valid_neighbors  # Center + 8 neighbors
            neighbor_indices.append(patch)

        return np.array(neighbor_indices)

    def _convert_to_data_indices(self):
        """Convert HEALPix cell IDs to data array indices"""
        data_indices = np.zeros_like(self.neighbor_indices)

        for i, patch in enumerate(self.neighbor_indices):
            for j, cell_id in enumerate(patch):
                data_indices[i, j] = self.cell_to_data_idx[cell_id]

        return torch.tensor(data_indices, dtype=torch.long)

    def forward(self, x):
        """Forward pass"""
        batch_size, n_channels, n_cells = x.shape

        # Ensure we have the right number of cells
        assert n_cells == len(self.available_cell_ids), \
            f"Expected {len(self.available_cell_ids)} cells, got {n_cells}"

        # Extract patches using the neighbor indices
        # Shape: [B, C_in, N_patches, 9]
        patches = x[:, :, self.data_neighbor_indices]

        # Reshape to [B, C_in, N_patches * 9] for Conv1d
        patches_flat = patches.view(batch_size, n_channels, -1)

        # Apply convolution
        output = self.conv(patches_flat)

        return output

class SphericalConvBlock(nn.Module):
    """Basic convolutional block with batch norm and ReLU"""

    def __init__(self, available_cell_ids, level, in_channels, out_channels, stride=1):
        super(SphericalConvBlock, self).__init__()

        self.conv = RegionalSphericalConv(
            available_cell_ids=available_cell_ids,
            level=level,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SphericalDoubleConv(nn.Module):
    """Double convolution block (Conv -> BN -> ReLU -> Conv -> BN -> ReLU)"""

    def __init__(self, available_cell_ids, level, in_channels, out_channels, stride=1):
        super(SphericalDoubleConv, self).__init__()

        # First conv with specified stride
        self.conv1 = SphericalConvBlock(
            available_cell_ids=available_cell_ids,
            level=level,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
        )

        # Second conv with stride=1 (operating on the output of first conv)
        self.conv2 = SphericalConvBlock(
            available_cell_ids=available_cell_ids,
            level=level,
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SimpleSphericalModel(nn.Module):
    """Simple model that just applies SphericalDoubleConv"""

    def __init__(self, available_cell_ids, level, in_channels, out_channels, stride=1):
        super(SimpleSphericalModel, self).__init__()

        self.double_conv = SphericalDoubleConv(
            available_cell_ids=available_cell_ids,
            level=level,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
        )

    def forward(self, x):
        return self.double_conv(x)


# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Your code with minimal fixes
ds_healpix = xr.open_dataset("/home/ubuntu/project/sentinel-2-dggs-ai-processor/src/notebook/healpix.zarr")

# 2. Get available cell IDs from your dataset
available_cell_ids = ds_healpix.cell_ids.values
print(f"Number of available HEALPix cells: {len(available_cell_ids)}")

# You need to define these variables
level = 19  # Your HEALPix level
band_list = ['b02', 'b03', 'b04', 'b08']  # Your spectral bands
in_channels = len(band_list)
stride = 1

model = SimpleSphericalModel(
    available_cell_ids=available_cell_ids,
    level=level,
    in_channels=in_channels,
    out_channels=in_channels,
    stride=stride
)

# Move model to GPU
model = model.to(device)

# Calculate total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"  - Total parameters: {total_params:,}")

# Create input tensor with all spectral bands
# Shape: [n_bands, n_cells]
spectral_data = []
for band in band_list:
    band_data = ds_healpix.Sentinel2.sel(bands=band).compute().values
    spectral_data.append(band_data)

# Stack all bands: [n_bands, n_cells]
x_multi_band = np.stack(spectral_data, axis=0)
print(f"Multi-band data shape: {x_multi_band.shape}")

# 5. Convert to PyTorch tensor and add batch dimension
x_tensor = torch.tensor(x_multi_band, dtype=torch.float32).unsqueeze(0)
print(f"Input tensor shape: {x_tensor.shape}")  # [1, n_bands, n_cells]

# Move input to GPU
x_tensor = x_tensor.to(device)

# 6. Forward pass
with torch.no_grad():
    output = model(x_tensor)  # Fixed: was SphericalUNet(x_tensor)
    print(f"Output tensor shape: {output.shape}")  # Fixed: was output.shape