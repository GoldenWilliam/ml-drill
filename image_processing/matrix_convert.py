from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define image paths for the three images you want to process
image_paths = ['output_image.png', 'output_image2.png', 'output_image3.png']

# Define the color mappings
color_to_number = {
    (255, 0, 0): 1,  # Red
    (0, 255, 0): 2,  # Green
    (0, 0, 255): 3  # Blue
}

# Initialize a list to store each resized matrix
all_resized_matrices = []

for image_path in image_paths:
    # Load the image and ensure it is in RGB format
    image = Image.open(image_path).convert('RGB')
    image_matrix = np.array(image)

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

    # Append the resized matrix to the list
    all_resized_matrices.append(resized_matrix)

    # Display the resized and interpolated image for each processed image
    plt.imshow(resized_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Color Code (1=Red, 2=Green, 3=Blue)')
    plt.title(f"Scaled Down and Interpolated Image (10x100) - {image_path}")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()

# Save all scaled-down matrices to a single text file
with open('scaled_down_matrices.txt', 'w') as f:
    for idx, resized_matrix in enumerate(all_resized_matrices):
        for row in resized_matrix:
            f.write(' '.join(map(str, row)) + '\n')
        f.write('\n')  # Add a blank line between matrices for readability

print("All scaled-down matrices saved to scaled_down_matrices.txt")


