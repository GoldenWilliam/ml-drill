import numpy as np
import gempy as gp
from generate_simulated_fields.generate_orientations import orientations
from generate_simulated_fields.generate_surface_points import surface_points


def generate_2D_field():
    orientations()
    surface_points()

    data_orientations = 'orientations.csv'
    data_surface_points = 'surface_points.csv'
    resolution =25

    geo_model: gp.data.GeoModel = gp.create_geomodel(
        project_name='Tutorial_ch1_1_Basics',
        extent=[0, 100, 0, 100, 0, 10],
        resolution=[100,100,10],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_orientations,
            path_to_surface_points=data_surface_points,
            hash_surface_points="4cdd54cd510cf345a583610585f2206a2936a05faaae05595b61febfc0191563",
            hash_orientations="7ba1de060fc8df668d411d0207a326bc94a6cdca9f5fe2ed511fd4db6b3f3526"
        )
    )

    #plot = gpv.plot_2d(geo_model, show_lith=True, show_boundaries=False)
    sol = gp.compute_model(geo_model)

    scalar_field = sol.raw_arrays.scalar_field_matrix[0]
    lith_type = sol.raw_arrays.lith_block
    lith_type = np.reshape(lith_type, (100, 100 , 10))




    htmp_3d = lith_type
    y_index = 1

    htmp_2d = htmp_3d[:, y_index, :]
    htmp_2d_rot = np.rot90(htmp_2d)

    return htmp_2d_rot