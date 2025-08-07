from numcodecs import Zstd
import xdggs
import xarray as xr
import numpy as np
from pyproj import Transformer
import healpy as hp

def get_bands(data_tree, res):

    res_key = f"r{res}"
    data = data_tree.measurements.reflectance[res_key]
    return list(data.keys())

def get_chunk_info(data_tree, band, res):
    """
    Extract chunk size and number of chunks from a dataset.

    Parameters:
    - data_tree: xarray.DataTree
    - band: str, e.g. "b03"
    - resolution: str, y-dimension name (e.g. "y_10m")
    - x_res: str, x-dimension name (e.g. "x_10m")

    Returns:
    - chunk_size_y: int
    - chunk_size_x: int
    - nb_chunks_y: int
    - nb_chunks_x: int
    """
    res_key = f"r{res}"
    y_res = f"y_{res}"
    x_res = f"x_{res}"
    data_tree = data_tree.measurements.reflectance[res_key]

    chunk_size_y = data_tree[band].chunksizes[y_res][0]
    chunk_size_x = data_tree[band].chunksizes[x_res][0]
    nb_chunks_y = len(data_tree[band].chunksizes[y_res])
    nb_chunks_x = len(data_tree[band].chunksizes[x_res])

    print(f"Chunk size: y={chunk_size_y}, x={chunk_size_x}")
    print(f"Number of chunks: y={nb_chunks_y}, x={nb_chunks_x}")

    return chunk_size_y, chunk_size_x, nb_chunks_y, nb_chunks_x

def get_chunk(data_tree, res, chunk_y_idx, chunk_x_idx, chunk_size_y, chunk_size_x):
    """
    Extract a specific chunk from a given band at a given spatial resolution in a DataTree.

    Parameters:
    - data_tree: xarray.DataTree
        The root DataTree object loaded from a Zarr store (e.g., xr.open_datatree(...)).
    - band: str
        The band name to extract (e.g., "b03").
    - res: str
        The spatial resolution as a string (e.g., "10m", "20m", "60m").
    - chunk_y_idx: int
        Index of the chunk along the vertical (y) axis.
    - chunk_x_idx: int
        Index of the chunk along the horizontal (x) axis.

    Returns:
    - xarray.DataArray
        A DataArray corresponding to the specified chunk.
    """
    res_key = f"r{res}"
    y_res = f"y_{res}"
    x_res = f"x_{res}"
    data = data_tree.measurements.reflectance[res_key]

    y_start = chunk_y_idx * chunk_size_y
    x_start = chunk_x_idx * chunk_size_x
    return data.isel(
        {y_res: slice(y_start, y_start + chunk_size_y),
         x_res: slice(x_start, x_start + chunk_size_x)}
    )


class proj_odysea:
    """
    HEALPix projection class for spatial data aggregation compatible with xdggs.

    This class performs spatial aggregation of Earth observation data onto HEALPix
    grids, enabling spherical analysis and processing. It transforms irregular
    spatial data into a hierarchical equal-area pixelization suitable for
    spherical deep learning and global analysis.

    The class aggregates pixel values within each HEALPix cell using mean averaging,
    handling missing data and maintaining compatibility with the xdggs ecosystem
    for further spherical data processing.

    Parameters
    ----------
    level : int
        HEALPix resolution level. NSIDE = 2^level determines the grid fineness.
        Higher levels provide finer spatial resolution.
    heal_idx : array-like
        Array of HEALPix cell indices covering the region of interest.
    inv_idx : array-like
        Inverse indices mapping each input pixel to its corresponding HEALPix cell.
    nscale : int, optional
        Scaling factor for processing. Default: 2.
    nest : bool, optional
        Whether to use nested HEALPix indexing scheme. Default: False (ring).
    chunk_size : int, optional
        Chunk size for dask array processing. Default: 4096.
    cell_id_name : str, optional
        Name for the cell ID coordinate. Default: "cell_ids".

    Attributes
    ----------
    level : int
        HEALPix resolution level.
    nside : int
        HEALPix NSIDE parameter (2^level).
    cell_ids : numpy.ndarray
        Flattened array of HEALPix cell IDs.
    var_cell_ids : xarray.DataArray
        Cell IDs as xarray coordinate with xdggs-compatible attributes.
    inv_idx : numpy.ndarray
        Flattened inverse indices for pixel-to-cell mapping.
    him : numpy.ndarray
        Histogram counts of pixels per HEALPix cell.

    Notes
    -----
    - Uses mean aggregation for multiple pixels within the same HEALPix cell
    - Handles missing data (NaN, zero values) appropriately
    - Maintains xdggs compatibility for downstream spherical processing
    - Supports both nested and ring HEALPix indexing schemes

    Examples
    --------
    >>> # Create HEALPix projection for level 10
    >>> level = 10
    >>> heal_idx = np.array([100, 101, 102, 103])  # HEALPix cell indices
    >>> inv_idx = np.array([0, 0, 1, 1, 2, 3])    # Pixel-to-cell mapping
    >>> proj = proj_odysea(level, heal_idx, inv_idx, nest=True)
    >>>
    >>> # Project dataset to HEALPix
    >>> ds_healpix = proj.eval(input_dataset)
    """

    def __init__(
        self,
        level,
        heal_idx,
        inv_idx,
        nscale=2,
        nest=False,
        chunk_size=4096,
        cell_id_name="cell_ids",
    ):
        self.level = level
        self.nside = 2**(level)
        self.nscale = nscale
        self.nest = nest
        self.chunk_size = chunk_size
        self.cell_id_name = cell_id_name

        # HEALPix cell setup with ONLY xdggs-compatible attributes
        self.cell_ids = heal_idx.flatten()
        self.var_cell_ids = xr.DataArray(
            self.cell_ids,
            dims="cells",
            attrs={
                "grid_name": "healpix",
                "indexing_scheme": "nested" if self.nest else "ring",
                "resolution": self.level,
                # Remove ALL legacy attributes that cause conflicts
            }
        )
        self.inv_idx = inv_idx.flatten()
        self.him = np.bincount(self.inv_idx)

    def eval(self, ds):
        """
        Convert dataset to HEALPix projection without time dimension.

        Aggregates spatial data from the input dataset onto HEALPix cells using
        mean averaging. Each HEALPix cell contains the average value of all
        pixels that fall within its boundaries, providing a spherically-aware
        representation of the original data.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset containing spatial variables to be projected.
            Expected to have data variables with spatial dimensions that can
            be flattened for processing.

        Returns
        -------
        xarray.Dataset
            HEALPix-projected dataset with dimensions:
            - 'bands': Data variables from input dataset
            - 'cells': HEALPix cell indices

            Contains:
            - 'Sentinel2': Main data array with aggregated values
            - Cell coordinates with xdggs-compatible attributes

        Notes
        -----
        Processing pipeline:
        1. Extracts all data variables from input dataset
        2. Flattens spatial dimensions for each variable
        3. Filters out invalid data (zeros, NaNs)
        4. Aggregates values within each HEALPix cell using mean
        5. Creates xdggs-compatible output structure
        6. Applies chunking for efficient processing

        The aggregation uses bincount with weights for efficient computation:
        - Sum of values per cell / count of pixels per cell = mean value
        - Cells with no valid pixels are assigned NaN

        Examples
        --------
        >>> # Project Sentinel-2 data to HEALPix
        >>> ds_input = xr.Dataset({
        ...     'B02': (['x', 'y'], reflectance_data_b02),
        ...     'B03': (['x', 'y'], reflectance_data_b03),
        ... })
        >>> ds_healpix = proj.eval(ds_input)
        >>> print(ds_healpix.dims)  # {'bands': 2, 'cells': n_healpix_cells}
        """
        var_name = list(ds.data_vars)
        print(f"Processing {len(var_name)} variables: {var_name}")

        # Initialize 2D data array (bands, cells)
        all_data = np.zeros([len(var_name), self.cell_ids.shape[0]])

        # Process each variable
        for i in range(len(var_name)):
            ivar = var_name[i]
            print(f"Processing {ivar} ({i+1}/{len(var_name)})")

            # Flatten spatial data
            b_data = ds[ivar].values.flatten()

            # Find valid data (non-zero and non-NaN)
            idx = np.where((b_data != 0) & (~np.isnan(b_data)))

            # Aggregate to HEALPix cells
            data = np.bincount(
                self.inv_idx[idx],
                weights=b_data[idx],
                minlength=self.cell_ids.shape[0]
            )

            # Count pixels per cell
            hdata = np.bincount(
                self.inv_idx[idx],
                minlength=self.cell_ids.shape[0]
            )

            # Calculate mean (handle division by zero)
            data = data.astype(float)
            data[hdata == 0] = np.nan
            valid_mask = hdata > 0
            data[valid_mask] = data[valid_mask] / hdata[valid_mask]

            # Store in 2D array
            all_data[i] = data

        # Create DataArray with correct dimensions
        data_array = xr.DataArray(
            all_data,
            dims=("bands", "cells"),
            coords={
                "bands": var_name,
                self.cell_id_name: self.var_cell_ids
            },
            name='Sentinel2',
            attrs={
                "description": "Sentinel-2 reflectance aggregated to HEALPix cells"
            }
        )

        # Convert to Dataset
        ds_total = data_array.to_dataset()

        # Set ONLY xdggs-compatible attributes (no extra attributes)
        ds_total[self.cell_id_name].attrs = {
            "grid_name": "healpix",
            "indexing_scheme": "nested" if self.nest else "ring",
            "resolution": self.level,
        }

        # Apply chunking
        chunk_size_data = max(1, int((12 * (4**self.level)) / self.chunk_size))
        ds_total = ds_total.chunk({"cells": chunk_size_data})

        print(f"HEALPix conversion complete - Level {self.level}, {len(self.cell_ids):,} cells")

        return ds_total


def healpix_projection(ds, utm_crs, level=19, chunk_size=4096):
    """
    Project Earth observation data from UTM coordinates to HEALPix spherical grid.

    This function performs a complete coordinate transformation and spatial aggregation
    pipeline, converting data from UTM projection to HEALPix equal-area spherical
    pixelization. It handles coordinate transformation, HEALPix index generation,
    and data aggregation in a single workflow.

    The pipeline enables spherical analysis of Earth observation data by:
    1. Transforming UTM coordinates to latitude/longitude
    2. Computing HEALPix cell indices for each pixel
    3. Aggregating data within HEALPix cells
    4. Creating xdggs-compatible output for spherical processing

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input dataset with UTM coordinates and spatial data variables.
        Must contain 'x' and 'y' coordinates and 'other_metadata' with CRS info.
    level : int, optional
        HEALPix resolution level (NSIDE = 2^level). Default: 19.
        Higher levels provide finer spatial resolution.
    chunk_size : int, optional
        Chunk size for dask array processing. Default: 4096.

    Returns
    -------
    xarray.Dataset
        HEALPix-projected dataset with xdggs encoding applied.
        Contains aggregated data with spherical coordinates and metadata
        compatible with spherical analysis tools.

    Notes
    -----
    Processing Pipeline:
    1. **Coordinate Grid Creation**: Generate meshgrid from UTM x,y coordinates
    2. **CRS Transformation**: Convert UTM to WGS84 lat/lon using pyproj
    3. **HEALPix Indexing**: Compute HEALPix cell indices using healpy
    4. **Data Aggregation**: Aggregate pixel values within each HEALPix cell
    5. **xdggs Encoding**: Apply xdggs decode for spherical data compatibility

    The function automatically:
    - Extracts CRS information from dataset metadata
    - Handles coordinate transformation with proper UTM zone detection
    - Uses nested HEALPix indexing for hierarchical processing
    - Maintains data integrity through mean aggregation

    Raises
    ------
    KeyError
        If required coordinates ('x', 'y') or metadata are missing from input.
    ValueError
        If coordinate transformation fails or invalid HEALPix level specified.

    Examples
    --------
    >>> # Project Sentinel-2 scene to HEALPix level 18
    >>> ds_sentinel = xr.open_dataset('sentinel2_scene.nc')
    >>> ds_healpix = healpix_projection(ds_sentinel, level=18)
    >>> print(f"Projected to {len(ds_healpix.cell_ids)} HEALPix cells")

    >>> # High-resolution projection for detailed analysis
    >>> ds_hires = healpix_projection(ds_sentinel, level=20, chunk_size=2048)

    See Also
    --------
    proj_odysea : Core HEALPix projection class
    xdggs.decode : Spherical grid decoding for analysis
    healpy.ang2pix : HEALPix index computation
    """

    x = ds["x"].values
    y = ds["y"].values
    xx, yy = np.meshgrid(x, y)

    print(f"Coordinate grid shape: {xx.shape}")
    print(f"X range: {x.min():.0f} to {x.max():.0f}")
    print(f"Y range: {y.min():.0f} to {y.max():.0f}")

    # 2. Transform UTM to lat/lon

    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)

    # 3. Generate HEALPix indices
    nside = 2 ** level
    idx = hp.ang2pix(nside, lon, lat, lonlat=True, nest=True)
    lidx, ilidx = np.unique(idx, return_inverse=True)

    print(f"HEALPix Level {level} → {len(lidx):,} unique cells")

    # 4. Project to HEALPix
    proj = proj_odysea(level, lidx, ilidx, nest=True, chunk_size=chunk_size)
    ds_healpix = proj.eval(ds.to_dataset())

    # # 5. Decode to add lat/lon if available via xdggs
    # if "xdggs" in globals():
    ds_healpix = ds_healpix.pipe(xdggs.decode)

    return ds_healpix