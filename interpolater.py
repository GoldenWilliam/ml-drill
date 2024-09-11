import numpy as np
from scipy.interpolate import griddata


def interpolate(x: np.array, y: np.array, values : np.array, grid_size: tuple[int,int],method='nearest'):
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
        return one_hole_interpolation(y,values,grid_size)
    
    grid_x, grid_y = np.meshgrid(np.linspace(0,grid_size[1],grid_size[1]),np.linspace(0,grid_size[0],grid_size[0]))
    points = np.array(list(zip(x, y)))

    ip_linaer = griddata(points,values, (grid_x,grid_y), method="linear")
    ip_nearest = griddata(points,values, (grid_x,grid_y), method="nearest")

    ip_field = np.where(np.isnan(ip_linaer),ip_nearest,ip_linaer)

    return ip_field
   
def one_hole_interpolation(y,values,grid_size):
    ip_field = np.empty(grid_size)

    for y_i in y:
        ip_field[int(y_i),:] = values

    return ip_field.T



