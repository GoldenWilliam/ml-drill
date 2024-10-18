import numpy as np
import matplotlib.pyplot as plt
from interpolator import interpolate, one_hole_interpolation

def test_interpolate():
    # Generate some random points and corresponding RGB values
    x = np.array([5, 10, 20, 30, 40])
    y = np.array([10, 20, 15, 25, 5])

    # Generate random RGB values for these points (flattened into a single array)
    values = np.array([144, 214, 67,    # RGB for point 1
                       255, 100, 50,    # RGB for point 2
                       200, 150, 100,   # RGB for point 3
                       90, 170, 200,    # RGB for point 4
                       120, 180, 220])  # RGB for point 5

    # Define the grid size (80 x 805 pixels as in your original data)
    grid_size = (80, 805)

    # Perform interpolation on the grid
    ip_field = interpolate(x, y, values, grid_size)

    # Print some debug info
    print(f"Interpolated field shape: {ip_field.shape}")
    print(f"Min value: {ip_field.min()}, Max value: {ip_field.max()}")

    # Clip the values to be within the valid range and cast to integers for displaying as an RGB image
    ip_field_clipped = np.clip(ip_field, 0, 255).astype(np.uint8)

    # Visualize the interpolated RGB field
    plt.imshow(ip_field_clipped, origin='lower')
    plt.title('Interpolated RGB Field')
    plt.show()

def test_one_hole_interpolation():
    # Single x-coordinate (testing the special case)
    x = np.array([10])
    y = np.array([5, 15, 25, 35, 45, 55])

    # Generate corresponding RGB values for each y-coordinate
    values = np.array([255, 0, 0,     # Red for point 1
                       0, 255, 0,     # Green for point 2
                       0, 0, 255,     # Blue for point 3
                       255, 255, 0,   # Yellow for point 4
                       255, 0, 255,   # Magenta for point 5
                       0, 255, 255])  # Cyan for point 6

    # Define the grid size (80 x 805 pixels as in your original data)
    grid_size = (80, 805)

    # Perform interpolation (though in this case it's more like assignment)
    ip_field = one_hole_interpolation(x, y, values, grid_size)

    # Print some debug info
    print(f"Interpolated field shape (one hole): {ip_field.shape}")
    print(f"Min value: {ip_field.min()}, Max value: {ip_field.max()}")

    # Clip the values to be within the valid range and cast to integers for displaying as an RGB image
    ip_field_clipped = np.clip(ip_field, 0, 255).astype(np.uint8)

    # Visualize the interpolated RGB field for the special case
    plt.imshow(ip_field_clipped, origin='lower')
    plt.title('One Hole Interpolated RGB Field')
    plt.show()

# Run the tests
test_interpolate()
test_one_hole_interpolation()
