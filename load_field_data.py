import numpy as np
import random
import re


class GetFields:
    """Class to handle loading data from arrays or files containing pixel data (e.g., RGB values)."""

    def __init__(self):
        self.training_data = []
        self.i = -1

    def load_data_from_file(self, filename: str):
        """Load data from a file containing RGB pixel data."""
        print(f"Attempting to load data from file: {filename}")
        try:
            with open(filename, 'r') as file:
                field = []
                for row in file:
                    print(f"Processing row: {row.strip()}")  # Debugging statement
                    # Use regex to find all groups of three numbers inside the brackets
                    rgb_values = re.findall(r'\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]', row)
                    for rgb in rgb_values:
                        array = [int(num) for num in rgb]  # Convert the extracted values to integers
                        if len(array) == 3:  # Ensure it's an RGB value (3 values per pixel)
                            field.append(array)

                if len(field) != 0:
                    self.training_data.append(np.asarray(field).reshape(-1, 3))  # Reshape into proper RGB format
                    print(f"Loaded field with {len(field)} pixels.")  # Debugging statement
                else:
                    print("Warning: No valid data found in the file.")  # Debugging statement

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found. Please check the file path.")

    def get_field(self):
        """Get the next field (dataset) from the training data."""
        if not self.training_data:
            raise ValueError("No training data loaded. Ensure that data is properly loaded before calling get_field.")

        self.i += 1
        if self.i >= len(self.training_data) or self.i == 0:
            self.i = 0
            random.shuffle(self.training_data)

        return np.asarray(self.training_data[self.i])