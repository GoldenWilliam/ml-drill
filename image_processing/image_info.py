from PIL import Image
import numpy as np

# Load the image
image_path = 'output_image.png'
image = Image.open(image_path).convert('RGB')  # Convert to RGB to ignore alpha channel if present

# Convert the image to an RGB matrix
image_matrix = np.array(image)

# Reshape the matrix to a list of RGB values (each pixel as an RGB triplet)
pixels = image_matrix.reshape(-1, image_matrix.shape[2])

# Find unique colors
unique_colors = np.unique(pixels, axis=0)

# Count the number of unique colors
num_unique_colors = unique_colors.shape[0]

print("Number of unique colors:", num_unique_colors)
print (image_matrix)
