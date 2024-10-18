import numpy as np
from scipy.interpolate import griddata


def interpolate(x: np.array, y: np.array, values: np.array, grid_size: tuple[int, int], method='linear'):
    """
    Interpolates data from hole points (x, y) with corresponding values, and returns the interpolated grid.
    """
    if len(np.unique(x)) < 2:  # If all x-values are the same, handle this case.
        print("All x values are the same, applying nearest-neighbor interpolation.")
        return np.full((grid_size[0], grid_size[1], 3), np.mean(values), dtype=np.uint8)

    grid_x, grid_y = np.meshgrid(np.linspace(0, grid_size[1] - 1, grid_size[1]),
                                 np.linspace(0, grid_size[0] - 1, grid_size[0]))

    points = np.array(list(zip(x, y)))

    # Perform interpolation for each color channel separately (RGB)
    r_values = values[:, 0]
    g_values = values[:, 1]
    b_values = values[:, 2]

    try:
        r_interp = griddata(points, r_values, (grid_x, grid_y), method=method, fill_value=0)
        g_interp = griddata(points, g_values, (grid_x, grid_y), method=method, fill_value=0)
        b_interp = griddata(points, b_values, (grid_x, grid_y), method=method, fill_value=0)

        ip_field = np.stack((r_interp, g_interp, b_interp), axis=-1)
        return np.clip(ip_field, 0, 255).astype(np.uint8)  # Ensure the values are valid RGB range

    except Exception as e:
        print(f"Interpolation error: {e}")
        # Fall back to nearest-neighbor interpolation
        return np.full((grid_size[0], grid_size[1], 3), np.mean(values), dtype=np.uint8)

def one_hole_interpolation(x, y, values, grid_size):
    """
    Handles interpolation when there's only one x-coordinate.

    Parameters:
    -----------
    x : np.array
        Array of x-coordinates for known points.
    y : np.array
        Array of y-coordinates for known points.
    values : np.array
        Array of RGB values corresponding to the (x, y) points (flattened array with size 3 times the number of points).
    grid_size : tuple[int, int]
        Size of the grid to interpolate on.

    Returns:
    --------
    np.ndarray
        Interpolated field.
    """
    ip_field = np.zeros((grid_size[0], grid_size[1], 3))  # 3D array to store RGB values

    # Separate the RGB channels from the values
    r_values = values[::3]  # R channel (every 3rd value starting from 0)
    g_values = values[1::3]  # G channel (every 3rd value starting from 1)
    b_values = values[2::3]  # B channel (every 3rd value starting from 2)

    # Assign values along the y dimension at the x position
    for i, y_i in enumerate(y):
        ip_field[int(y_i), :, 0] = r_values[i]  # Assign R values
        ip_field[int(y_i), :, 1] = g_values[i]  # Assign G values
        ip_field[int(y_i), :, 2] = b_values[i]  # Assign B values

    return ip_field