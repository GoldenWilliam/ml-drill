from fileinput import filename

import numpy as np
import gempy as gp
import gempy_viewer as gpv
import matplotlib.pyplot as plt
from plot_storage import save_incrementing_plot
import cv2 as cv

def generate_2D_field(orientations_file: str, surface_points_file: str, resolution_XYZ: list, show_gempy_plot: bool=False, show_regular_plot: bool=False):

    data_orientations = orientations_file
    data_surface_points = surface_points_file

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Boreholes',
        extent=[0, 100, 0, 100, 0, 10],
        resolution=[resolution_XYZ[0], resolution_XYZ[1], resolution_XYZ[2]],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_orientations,
            path_to_surface_points=data_surface_points,
            hash_surface_points="4cdd54cd510cf345a583610585f2206a2936a05faaae05595b61febfc0191563",
            hash_orientations="7ba1de060fc8df668d411d0207a326bc94a6cdca9f5fe2ed511fd4db6b3f3526"
        )
    )

    sol = gp.compute_model(geo_model)

    lith_type = sol.raw_arrays.lith_block
    lith_type = np.reshape(lith_type, (resolution_XYZ[0], resolution_XYZ[1], resolution_XYZ[2]))

    if show_gempy_plot:
        gpv.plot_2d(geo_model, show_data=False, legend=False, show_boundaries=False, cell_number=[1], direction='y')
        plt.show()

    htmp_3d = lith_type
    y_index = 1

    htmp_2d = htmp_3d[:, y_index, :]
    htmp_2d_rot = np.rot90(htmp_2d)

    filename = save_incrementing_plot(htmp_2d_rot, "soil_profile", show_regular_plot, "soil_profiles")

    image = cv.imread(filename)

    # Convert BGR to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Flip the image upside-down
    soil_matrix_rgb = np.flipud(image_rgb)

    return soil_matrix_rgb

# env2D = generate_2D_field(
#     orientations_file='orientations.csv',
#     surface_points_file='surface_points.csv',
#     resolution_XYZ=[25, 25, 25],
#     show_gempy_plot=True,
#     show_regular_plot=True)
#
# # Define the x-range of interest
# x_min, x_max = 60, 70
# n_cols = env2D.shape[1]
#
# # Calculate the column indices
# col_min = int((x_min / 100) * n_cols)
# col_max = int((x_max / 100) * n_cols)
#
# # Create a masked array: data between col_min and col_max stays, the rest is masked
# masked_htmp_2d = ma.masked_array(env2D, mask=np.ones_like(env2D))
# masked_htmp_2d[:, col_min:col_max] = env2D[:, col_min:col_max]
#
# cbar_indices = [0, 1, 2, 3, 4, 5]
#
# #Borehole plot
# plt.figure()
# plt.imshow(masked_htmp_2d, cmap='Set1', interpolation='nearest', extent=[0, 100, 0, 10])
# plt.colorbar(label='IC', boundaries=cbar_indices, ticks=cbar_indices)
# plt.xlabel('X[m]')
# plt.ylabel('Depth[m]')
# plt.title(f"Data Between x={x_min} and x={x_max}")
# plt.grid(True)
# plt.xticks(np.arange(0, 101, 5))
# plt.yticks(np.arange(0, 11, 5))
# plt.show()

# Old

# def generate_2D_field(orientations_file: str, surface_points_file: str, resolution_XYZ: list, show_gempy_plot: bool=False, show_regular_plot: bool=False):
#
#     data_orientations = orientations_file
#     data_surface_points = surface_points_file
#
#     geo_model: gp.data.GeoModel = gp.create_geomodel(
#         project_name='Boreholes',
#         extent=[0, 100, 0, 100, 0, 10],
#         resolution=[resolution_XYZ[0], resolution_XYZ[1], resolution_XYZ[2]],
#         importer_helper=gp.data.ImporterHelper(
#             path_to_orientations=data_orientations,
#             path_to_surface_points=data_surface_points,
#             hash_surface_points="4cdd54cd510cf345a583610585f2206a2936a05faaae05595b61febfc0191563",
#             hash_orientations="7ba1de060fc8df668d411d0207a326bc94a6cdca9f5fe2ed511fd4db6b3f3526"
#         )
#     )
#
#     sol = gp.compute_model(geo_model)
#
#     lith_type = sol.raw_arrays.lith_block
#     lith_type = np.reshape(lith_type, (resolution_XYZ[0], resolution_XYZ[1], resolution_XYZ[2]))
#
#     if show_gempy_plot:
#         gpv.plot_2d(geo_model, show_data=False, legend=False, show_boundaries=False, cell_number=[1], direction='y')
#         plt.show()
#
#     htmp_3d = lith_type
#     y_index = 1
#
#     htmp_2d = htmp_3d[:, y_index, :]
#     htmp_2d_rot = np.rot90(htmp_2d)
#     vmin = 0
#     vmax = 6
#     cbar_indices = [0, 1, 2, 3, 4, 5]
#
#     # plt.imshow(htmp_2d_rot, cmap='viridis', interpolation='nearest', extent=[0, 100,0, 10])
#     plt.imshow(htmp_2d_rot, cmap='viridis', origin='upper', interpolation='nearest', extent=[0, 100, 0, 10], vmin=vmin,
#                vmax=vmax)
#     # plt.colorbar(label='IC', boundaries=cbar_indices, ticks=cbar_indices)
#     # plt.xlabel('X[m]')
#     # plt.ylabel('Depth[m]')
#     plt.grid(False)
#     plt.axis('off')
#     # plt.xticks(np.arange(0, 101, 10))
#     # plt.yticks(np.arange(0, 11, 5))
#     plt.savefig('soil_profile.png', bbox_inches='tight', pad_inches=0)
#
#     if show_regular_plot:
#         plt.show()
#
#     return htmp_2d_rot