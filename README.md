# DLCV
## lab 1 transformation
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

[lab2](https://colab.research.google.com/drive/1z1wwZiJIDxoprMDtsNI07cDBbgjOGoOU?authuser=1&usp=classroom_web#scrollTo=7FHJN1zI3U98)
## lab 3 canny edge detection
[lab3](https://colab.research.google.com/drive/1iati55manaTaFjbolDmab2bnb3mojhEo?authuser=1&usp=classroom_web)
## lab 4 SIFT detector
[lab4](https://colab.research.google.com/drive/1FBQ6oOohWqO95H7cd6HawAOC7jvgGSz1?authuser=1&usp=classroom_web)
## lab 5 stitching panorama
[lab5](https://colab.research.google.com/drive/1wtcgOciMPhPn7hJvFlL4wWZDpV1DUmSq?authuser=1&usp=classroom_web)
## lab 6 BOW
[lab6](https://colab.research.google.com/drive/1Doz8T62kLWq4n_wjwJaQF1ZQhNl7WDa_?authuser=1&usp=classroom_web#scrollTo=Qq-9rtxZ7YlP)
