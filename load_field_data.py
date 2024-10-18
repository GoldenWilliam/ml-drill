import numpy as np
import re


class GetFields:
    """Class to handle loading data from arrays or files containing multiple models of pixel data (e.g., RGB values)."""

    def __init__(self):
        self.training_data = []
        self.i = -1

    def load_data_from_file(self, filename: str):
        """Load data from a file containing RGB pixel data, with 10 models per file."""
        print(f"Attempting to load data from file: {filename}")

        # Clear the previous data before loading a new file
        self.training_data = []
        self.i = -1  # Reset the index as well

        try:
            with open(filename, 'r') as file:
                model = []
                for row in file:
                    # Use regex to find all groups of three numbers inside the brackets (RGB values)
                    rgb_values = re.findall(r'\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]', row)
                    if rgb_values:
                        model.append([list(map(int, rgb)) for rgb in rgb_values])

                    # Assume a blank line separates models
                    if not row.strip() and model:
                        self.training_data.append(np.array(model))
                        model = []  # Reset for the next model

                if model:  # If the last model hasn't been appended
                    self.training_data.append(np.array(model))
                print(f"Loaded {len(self.training_data)} models.")

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found. Please check the file path.")

    def get_field(self):
        """Get the next field (model) from the training data."""
        if not self.training_data:
            raise ValueError("No training data loaded. Ensure that data is properly loaded before calling get_field.")

        self.i += 1
        if self.i >= len(self.training_data):
            self.i = 0
            np.random.shuffle(self.training_data)

        return np.asarray(self.training_data[self.i])
