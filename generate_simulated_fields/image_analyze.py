import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load the image using OpenCV
image = cv.imread('soil_profiles/soil_profile_1.png')

# Convert BGR to RGB
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Flip the image upside-down
soil_matrix_rgb = np.flipud(image_rgb)


# Plot the flipped image
plt.imshow(soil_matrix_rgb, cmap='viridis', origin='lower', extent=[0, 100,0, 10])
plt.axis('off')  # Hide axis
plt.savefig('hey2.png', bbox_inches='tight', pad_inches=0)
plt.show()

p = cv.imread('hey2.png')

if soil_matrix_rgb.shape == image.shape:
    print(True)
else:
    print(False)
print(soil_matrix_rgb.shape)
print(image.shape)

print(type(soil_matrix_rgb))