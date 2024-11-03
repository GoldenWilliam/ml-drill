from PIL import Image
import numpy as np

# Load the image and ensure it is in RGB format
image_path = 'images/ekte_data_rgb2.png'
image = Image.open(image_path).convert('RGB')
image_matrix = np.array(image)

# Define the target colors
red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]
black = [0, 0, 0]

# Iterate over each pixel and replace based on conditions
for i in range(image_matrix.shape[0]):
    for j in range(image_matrix.shape[1]):
        r, g, b = image_matrix[i, j]

        # Condition for red
        if r > 80 and g < 200 and b < 200:
            image_matrix[i, j] = red

        # Condition for green
        elif r < 200 and g > 80 and b < 200:
            image_matrix[i, j] = green

        # Condition for blue
        elif r < 200 and g < 200 and b > 80:
            image_matrix[i, j] = blue

        # All other colors to black
        else:
            image_matrix[i, j] = blue

# Convert the modified matrix back to an image
new_image = Image.fromarray(image_matrix.astype('uint8'))

# Save or display the new image
new_image.save('output_image2.png')
new_image.show()
