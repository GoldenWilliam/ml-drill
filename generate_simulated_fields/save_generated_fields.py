from sim2D import generate_2D_field
from generate_orientations import generate_orientations
from generate_surface_points import generate_surface_points
import os

# Create the directory for training data if it doesn't exist
os.makedirs("test_data1", exist_ok=True)

# Generate 800 files, each containing 10 fields
num_files = 1
fields_per_file = 1

for file_idx in range(1, num_files + 1):                               # Iterate through file numbers (1 to 80)
    # Create a filename for each file, using an iterating number
    filename = f"test_data/test_data_10_{file_idx}.txt"

    # Open the file to store the generated data for 10 fields
    with open(filename, 'w') as file:
        for _ in range(fields_per_file):  # Generate 10 fields per file
            # Generate orientations and surface points
            generate_orientations()
            generate_surface_points()

            # Generate the 2D field based on the generated orientations and surface points
            field = generate_2D_field(
                orientations_file='orientations.csv',
                surface_points_file='surface_points.csv',
                resolution_XYZ=[30, 30, 30],
                # show_gempy_plot=True,
                # show_regular_plot=True
            )

            # Convert the 2D field into a string that is suitable for writing to a file
            for row in field:
                # Convert each row to a space-separated string and write it to the file
                row_string = ' '.join(map(str, row))
                file.write(row_string + '\n')

            # Add a separator between fields for clarity
            file.write("\n\n")

    print(f"File {filename} generated successfully.")

# After the loop, all 80 files, each with 10 fields, will be created.