import numpy as np
from scipy.interpolate import griddata


def interpolate(x: np.array, y: np.array, values: np.array, grid_size: tuple[int, int], method='linear'):
    """
    Interpolates data from hole points (x, y) with corresponding values (RGB), and returns the interpolated grid.
    This function ensures the entire grid is filled by applying nearest-neighbor interpolation for areas
    where linear interpolation is not possible.
    """
    if len(np.unique(x)) < 2:  # If all x-values are the same, handle this case.
        print("All x values are the same, applying nearest-neighbor interpolation.")
        return np.full((grid_size[0], grid_size[1], 3), np.mean(values, axis=0), dtype=np.uint8)

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(np.linspace(0, grid_size[1] - 1, grid_size[1]),
                                 np.linspace(0, grid_size[0] - 1, grid_size[0]))

    points = np.array(list(zip(x, y)))

    # Perform interpolation for each color channel separately (RGB)
    r_values = values[:, 0]
    g_values = values[:, 1]
    b_values = values[:, 2]

    # Interpolate using the specified method (e.g., 'linear')
    r_interp = griddata(points, r_values, (grid_x, grid_y), method=method, fill_value=np.nan)
    g_interp = griddata(points, g_values, (grid_x, grid_y), method=method, fill_value=np.nan)
    b_interp = griddata(points, b_values, (grid_x, grid_y), method=method, fill_value=np.nan)

    # Stack the RGB channels back into a 3D array
    ip_field = np.stack((r_interp, g_interp, b_interp), axis=-1)

    # Now, handle NaN values by filling them with nearest neighbor interpolation
    nan_mask = np.isnan(ip_field)

    if np.any(nan_mask):
        # Apply nearest neighbor interpolation to fill NaN values
        r_interp_nn = griddata(points, r_values, (grid_x, grid_y), method='nearest')
        g_interp_nn = griddata(points, g_values, (grid_x, grid_y), method='nearest')
        b_interp_nn = griddata(points, b_values, (grid_x, grid_y), method='nearest')

        # Fill NaN values with nearest neighbor results
        ip_field[..., 0][nan_mask[..., 0]] = r_interp_nn[nan_mask[..., 0]]
        ip_field[..., 1][nan_mask[..., 1]] = g_interp_nn[nan_mask[..., 1]]
        ip_field[..., 2][nan_mask[..., 2]] = b_interp_nn[nan_mask[..., 2]]

    # Ensure that values are in the valid range for RGB (0-255)
    return np.clip(ip_field, 0, 255).astype(np.uint8)
