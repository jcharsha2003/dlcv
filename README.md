# DLCV
## lab 1 transformation
### WRITE CODE HERE ###
import io
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('tree_image.png')

# 2D translation
tx, ty = 50, 30 # Translation in x and y
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

# 2D rotation
angle = 45 # Rotation angle in degrees
rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# 2D scaling
scale_factor = 0.5
scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

# Display the transformed images
plt.figure(figsize=(12, 4))
plt.subplot(221), plt.imshow(translated_image), plt.title('Translated Image')
plt.subplot(222), plt.imshow(rotated_image), plt.title('Rotated Image')
plt.subplot(223), plt.imshow(scaled_image), plt.title('Scaled Image')
plt.subplot(224), plt.imshow(image), plt.title('Original Image')
plt.tight_layout()
plt.show()

# Load the image
image = cv2.imread('cube_image.png')
pts1 = np.float32([[10, 10], [100, 10], [10, 100], [100, 100]])
pts2 = np.float32([[0, 0], [150, 0], [0, 150], [150, 150]])

# Calculate the perspective transformation matrix
perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the perspective transformation
perspective_image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1],
image.shape[0]))

# Simulate 3D rotation
angle = 30 # Rotation angle in degrees
rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

# Display the transformed images
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB)), plt.title('Perspective Image')
plt.subplot(132), plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)), plt.title('Rotated Image')
plt.subplot(133), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.tight_layout()
plt.show()
[lab1](https://colab.research.google.com/drive/1u1DqUqinbUUJ580kvJlWrUUM5mT8QpmC?authuser=1&usp=classroom_web#scrollTo=MLd4d_hzJyAQ)
## lab 2 linear or non liner
### WRITE CODE HERE ###
import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_linear_filter(image, kernel):
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def apply_nonlinear_filter(image, filter_type):
    if filter_type == "median":
        filtered_image = cv2.medianBlur(image, 5)
    elif filter_type == "gaussian":
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "bilateral":
        filtered_image = cv2.bilateralFilter(image,15,75,75)
    elif filter_type == "min":
        filtered_image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    elif filter_type == "max":
        filtered_image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    else:
        raise ValueError("Invalid non-linear filter type.")
    return filtered_image

image_path = "./virat-kohli.jpg"
image = cv2.imread(image_path)

fig, axs = plt.subplots(1,3,figsize=(10,5))

# Display the original image
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")

# Box Filter 5X5
kernel = np.ones((5, 5), np.float32) / 25
linear_filtered_image = apply_linear_filter(image, kernel)
axs[1].imshow(cv2.cvtColor(linear_filtered_image, cv2.COLOR_BGR2RGB))
axs[1].set_title("Box Filter")

# Gausian Filter
gaussian_filtered_image = apply_nonlinear_filter(image, "gaussian")
axs[2].imshow(cv2.cvtColor(gaussian_filtered_image, cv2.COLOR_BGR2RGB))
axs[2].set_title("Gaussian Filter")

plt.tight_layout()
plt.show()
fig, axs = plt.subplots(2,3,figsize=(10,10))

# Display the original image
axs[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0,0].set_title("Original Image")

# Min Filter 5 x 5
min_filtered_image = apply_nonlinear_filter(image, "min")
axs[0,1].imshow(cv2.cvtColor(min_filtered_image, cv2.COLOR_BGR2RGB))
axs[0,1].set_title("Min Filter")

# Max Filter 5 x 5
max_filtered_image = apply_nonlinear_filter(image, "max")
axs[0,2].imshow(cv2.cvtColor(max_filtered_image, cv2.COLOR_BGR2RGB))
axs[0,2].set_title("Max Filter")

# Medain Filter 5 x 5
median_filtered_image = apply_nonlinear_filter(image, "median")
axs[1,0].imshow(cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2RGB))
axs[1,0].set_title("Median Filter")

# Biltaeral Filter
bil_filtered_image = apply_nonlinear_filter(image, "bilateral")
axs[1,1].imshow(cv2.cvtColor(bil_filtered_image, cv2.COLOR_BGR2RGB))
axs[1,1].set_title("Bilateral Filter")

axs[1,2].axis('off')
plt.tight_layout()
plt.show()
[lab2](https://colab.research.google.com/drive/1z1wwZiJIDxoprMDtsNI07cDBbgjOGoOU?authuser=1&usp=classroom_web#scrollTo=7FHJN1zI3U98)
## lab 3 canny edge detection
[lab3](https://colab.research.google.com/drive/1iati55manaTaFjbolDmab2bnb3mojhEo?authuser=1&usp=classroom_web)
## lab 4 SIFT detector
[lab4](https://colab.research.google.com/drive/1FBQ6oOohWqO95H7cd6HawAOC7jvgGSz1?authuser=1&usp=classroom_web)
## lab 5 stitching panorama
[lab5](https://colab.research.google.com/drive/1wtcgOciMPhPn7hJvFlL4wWZDpV1DUmSq?authuser=1&usp=classroom_web)
## lab 6 BOW
[lab6](https://colab.research.google.com/drive/1Doz8T62kLWq4n_wjwJaQF1ZQhNl7WDa_?authuser=1&usp=classroom_web#scrollTo=Qq-9rtxZ7YlP)
