import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import healpy as hp


class RegionalSphericalConv(nn.Module):
    def __init__(self, available_cell_ids, level, in_channels, out_channels, bias=True, nest=True, stride=1):
        """
        Regional Spherical Convolutional layer for HEALPix data covering a specific area.

        Parameters
        ----------
        available_cell_ids : array-like
            List/array of HEALPix cell IDs that are present in your dataset
        level : int
            HEALPix resolution level (NSIDE = 2^level)
        in_channels : int
            Number of input channels (e.g., spectral bands)
        out_channels : int
            Number of output channels
        bias : bool, optional
            Add bias term, by default True
        nest : bool, optional
            Use nested indexing, by default True
        stride : int, optional
            Stride for sampling center cells, by default 1
        """
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
        self.data_neighbor_indices = self._convert_to_data_indices()

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
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C_in, N] where:
            - B: batch size
            - C_in: number of input channels
            - N: number of available HEALPix cells

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, C_out, N_patches] where:
            - N_patches: number of valid patches (depends on stride)
        """
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



