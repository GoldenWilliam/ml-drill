from PIL import Image
import numpy as np

# Load the image
image_path = 'images/ekte_data_rgb.png'
image = Image.open(image_path).convert('RGB')  # Convert to RGB to ignore alpha channel if present

# Convert the image to an RGB matrix
image_matrix = np.array(image)

# Display the shape of the image matrix
print("Image Matrix Shape:", image_matrix.shape)

# Show the matrix
print("Image Matrix:\n", image_matrix)
