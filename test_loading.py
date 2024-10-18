import numpy as np
import matplotlib.pyplot as plt
from load_field_data import GetFields
import random

def display_all_fields(data_file):
    # Initialize the GetFields object and load data from the file
    data = GetFields()
    data.load_data_from_file(data_file)

    # Loop through all the fields in the loaded data
    num_fields = len(data.training_data)  # Assuming training_data contains all the models

    for i in range(num_fields):
        # Get the next field (it will iterate through the shuffled models)
        field = data.get_field()

        # Display the image
        plt.imshow(field.astype(np.uint8), origin='lower')
        plt.title(f"Field Image {i+1}")
        plt.axis("off")  # Hide the axis
        plt.show()

# Example usage: Change the path to your actual file
#display_all_fields('generate_simulated_fields/training_data/test_data_10_1.txt')

def display_all_fields_random(data_file):
    # Initialize the GetFields object and load data from the file
    data = GetFields()
    data.load_data_from_file(data_file)

    # Shuffle the indices for random display
    num_fields = len(data.training_data)
    indices = list(range(num_fields))
    random.shuffle(indices)

    # Loop through all the fields in a random order
    for i in indices:
        # Get the field based on the shuffled index
        field = data.training_data[i]
        print(type(field))

        # Display the image
        plt.imshow(field.astype(np.uint8), origin='lower')
        plt.title(f"Random Field Image {i+1}")
        plt.axis("off")  # Hide the axis
        plt.show()

# Example usage: Change the path to your actual file
display_all_fields_random('generate_simulated_fields/training_data/test_data_10_1.txt')