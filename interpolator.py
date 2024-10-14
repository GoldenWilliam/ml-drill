import numpy as np
from scipy.interpolate import griddata


def interpolate(x: np.array, y: np.array, values: np.array, grid_size: tuple[int, int], method='nearest'):
    """
    Interpolates data from hole points (x, y) with corresponding values, and returns the interpolated grid.

    Parameters:
    -----------
    x : np.array
        Array of x-coordinates for known points.
    y : np.array
        Array of y-coordinates for known points.
    values : np.array
        Array of values corresponding to the (x, y) points.
    grid_size : tuple[int, int]
        Size of the grid to interpolate on. It defines the dimensions of the output interpolated field.
    method : str, optional
        The interpolation method to use. Can be 'linear', 'nearest', or 'cubic'. Default is 'linear'.

    Returns:
    --------
    np.ndarray
        A 2D array representing the interpolated values over the specified grid.
    """
    # If we only have one x-coordinate
    if len(set(x)) == 1:
        return one_hole_interpolation(x, y, values, grid_size)

    grid_x, grid_y = np.meshgrid(np.linspace(0, grid_size[1] - 1, grid_size[1]),
                                 np.linspace(0, grid_size[0] - 1, grid_size[0]))
    points = np.array(list(zip(x, y)))

    # Perform interpolation
    ip_linear = griddata(points, values, (grid_x, grid_y), method="linear")
    ip_nearest = griddata(points, values, (grid_x, grid_y), method="nearest")

    # Fill NaN values in linear interpolation using nearest neighbor
    ip_field = np.where(np.isnan(ip_linear), ip_nearest, ip_linear)

    return ip_field


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
        Array of values corresponding to the (x, y) points.
    grid_size : tuple[int, int]
        Size of the grid to interpolate on.

    Returns:
    --------
    np.ndarray
        Interpolated field.
    """
    ip_field = np.zeros(grid_size)

    # Assign values along the y dimension at the x position
    for i, y_i in enumerate(y):
        ip_field[int(y_i), :] = values[i]  # Ensure that values are assigned row-wise

    return ip_field