import matplotlib.pyplot as plt
import numpy as np
from sim2D import generate_2D_field
from generate_orientations import orientations
from generate_surface_points import surface_points

with open("train_data_200_6.txt", 'w') as file:
    for _ in range(200):
        field = generate_2D_field(
            orientations_file=orientations(),
            surface_points_file=surface_points(),
            resolution_XYZ=[25, 25, 25])
        field_string = '\n'.join([''.join(map(str, row)) for row in field])

        file.write(field_string)
        file.write("\n\n")

file.close()