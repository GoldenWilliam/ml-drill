from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image and ensure it is in RGB format
image_path = 'output_image.png'  # Use the processed image file
image = Image.open(image_path).convert('RGB')
image_matrix = np.array(image)

# Define the color mappings
color_to_number = {
    (255, 0, 0): 1,       # Red
    (0, 255, 0): 2,       # Green
    (255, 255, 0): 3      # Yellow
}

# Initialize a matrix to store the numerical representation
number_matrix = np.zeros((image_matrix.shape[0], image_matrix.shape[1]), dtype=int)

# Map each pixel to a number based on color
for i in range(image_matrix.shape[0]):
    for j in range(image_matrix.shape[1]):
        pixel = tuple(image_matrix[i, j])
        number_matrix[i, j] = color_to_number.get(pixel, 0)  # 0 for unmatched colors

# Convert the number matrix to an image for resizing
number_image = Image.fromarray(number_matrix.astype('uint8'))

# Resize the image to 10x100 with interpolation
resized_image = number_image.resize((100, 10), Image.NEAREST)
resized_matrix = np.array(resized_image)

# Display the resized and interpolated image
plt.imshow(resized_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Color Code (1=Red, 2=Green, 3=Yellow)')
plt.title("Scaled Down and Interpolated Image (10x100)")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

# Save the scaled-down matrix to a text file
with open('scaled_down_matrix.txt', 'w') as f:
    for row in resized_matrix:
        f.write(' '.join(map(str, row)) + '\n')

print("Scaled-down matrix saved to scaled_down_matrix.txt")


