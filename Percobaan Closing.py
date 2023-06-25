import cv2
import numpy as np
from skimage import data
from skimage.io import imread

import matplotlib.pyplot as plt
#image = data.retina()
#image = data.astronaut()
image = imread(fname="aqua2.jpg")

print(image.shape)
plt.imshow(image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# defining the range of masking
blue1 = np.array([110, 50, 50])
blue2 = np.array([130, 255, 255])

# initializing the mask to be
# convoluted over input image
mask = cv2.inRange(hsv, blue1, blue2)

# passing the bitwise_and over
# each pixel convoluted
res = cv2.bitwise_and(image, image, mask=mask)

# defining the kernel i.e. Structuring element
kernel = np.ones((5, 5), np.uint8)

# defining the closing function
# over the image and structuring element
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
fig, axes = plt.subplots(1, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(mask)
ax[0].set_title("Citra Input 1")

ax[1].imshow(closing, cmap='gray')
ax[1].set_title('Citra Input 2')
plt.show()
# organizing imports
import cv2
import numpy as np

screenRead = cv2.VideoCapture(0)

while (1):
    _, image = screenRead.read()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue1 = np.array([110, 50, 50])
    blue2 = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, blue1, blue2)

    res = cv2.bitwise_and(image, image, mask=mask)

    kernel = np.ones((5, 5), np.uint8)

    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Mask', mask)
    cv2.imshow('Closing', closing)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cv2.destroyAllWindows()

screenRead.release()
plt.show()