import cv2
import matplotlib.pyplot as plt
import numpy as np

im_src = cv2.imread("example.jpg")

im = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB))
plt.show()

edges = cv2.Canny(im, 220, 105)

# plt.imshow(edges, cmap='gray')
# plt.show()

# HOW IS THIS NOT GETTING ALL THE PIXELS
_, binary = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
mask = binary
# kernel = np.ones((3, 3), np.uint8)
# im1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(binary, contours, -1, (255, 255, 255))

plt.subplot(121), plt.imshow(mask, cmap="gray")
plt.subplot(122), plt.imshow(binary, cmap='gray')
plt.show()
