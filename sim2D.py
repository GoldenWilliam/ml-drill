import numpy as np
import gempy as gp
import gempy_viewer as gpv
import matplotlib.pyplot as plt
from numpy import ma  # For masking


def import_2D_env(orientations_file: str, surface_points_file: str, resolution: int, show_gempy_plot: bool=False, show_regular_plot: bool=False):

    data_orientations = orientations_file
    data_surface_points = surface_points_file

    res = resolution

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Boreholes',
        extent=[0, 100, 0, 100, 0, 10],
        resolution=[res,res,res],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_orientations,
            path_to_surface_points=data_surface_points,
            hash_surface_points="4cdd54cd510cf345a583610585f2206a2936a05faaae05595b61febfc0191563",
            hash_orientations="7ba1de060fc8df668d411d0207a326bc94a6cdca9f5fe2ed511fd4db6b3f3526"
        )
    )

    sol = gp.compute_model(geo_model)

    lith_type = sol.raw_arrays.lith_block
    lith_type = np.reshape(lith_type, (res, res, res))

    if show_gempy_plot:
        gpv.plot_2d(geo_model, show_data=False, legend=False, show_boundaries=False, cell_number=[1], direction='y')
        plt.show()

    htmp_3d = lith_type
    y_index = 1

    htmp_2d = htmp_3d[:, y_index, :]
    htmp_2d_rot = np.rot90(htmp_2d)

    if show_regular_plot:
        cbar_indices = [0, 1, 2, 3, 4, 5]

        plt.imshow(htmp_2d_rot, cmap='Set1', interpolation='nearest', extent=[0, 100, 0, 10])
        plt.colorbar(label='IC', boundaries=cbar_indices, ticks=cbar_indices)
        plt.xlabel('X[m]')
        plt.ylabel('Depth[m]')
        plt.grid(True)
        plt.xticks(np.arange(0, 101, 10))
        plt.yticks(np.arange(0, 11, 5))
        plt.show()

    return htmp_2d_rot

env2D = import_2D_env(orientations_file='orientations.csv', surface_points_file='surface_points.csv', resolution=25, show_gempy_plot=True, show_regular_plot=True)

# Define the x-range of interest
x_min, x_max = 60, 70
n_cols = env2D.shape[1]

# Calculate the column indices
col_min = int((x_min / 100) * n_cols)
col_max = int((x_max / 100) * n_cols)

# Create a masked array: data between col_min and col_max stays, the rest is masked
masked_htmp_2d = ma.masked_array(env2D, mask=np.ones_like(env2D))
masked_htmp_2d[:, col_min:col_max] = env2D[:, col_min:col_max]

cbar_indices = [0, 1, 2, 3, 4, 5]

#Borehole plot
plt.figure()
plt.imshow(masked_htmp_2d, cmap='Set1', interpolation='nearest', extent=[0, 100, 0, 10])
plt.colorbar(label='IC', boundaries=cbar_indices, ticks=cbar_indices)
plt.xlabel('X[m]')
plt.ylabel('Depth[m]')
plt.title(f"Data Between x={x_min} and x={x_max}")
plt.grid(True)
plt.xticks(np.arange(0, 101, 5))
plt.yticks(np.arange(0, 11, 5))
plt.show()

print(masked_htmp_2d)