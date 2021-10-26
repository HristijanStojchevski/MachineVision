import cv2
import matplotlib.pyplot as plt
import numpy as np

im_src = cv2.imread("example.jpg")

im = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB))
plt.show()

# Basic thresholding

_, binary = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)  # THIS
_, binaryinv = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY_INV)
tr, binaryotsu = cv2.threshold(im, 128, 255, cv2.THRESH_OTSU)  # THIS
_, binarytozero = cv2.threshold(im, 128, 255, cv2.THRESH_TOZERO)
_, binarytozeroinv = cv2.threshold(im, 128, 255, cv2.THRESH_TOZERO_INV)
_, binarytrunc = cv2.threshold(im, 128, 255, cv2.THRESH_TRUNC)

plt.imshow(binary, cmap="gray")
plt.show()
# plt.imshow(binaryinv, cmap="gray")
# plt.show()
# plt.imshow(binaryotsu, cmap="gray")
# plt.show()
# plt.imshow(binarytozero, cmap="gray")
# plt.show()
# plt.imshow(binarytozeroinv, cmap="gray")
# plt.show()
# plt.imshow(binarytrunc, cmap="gray")
# plt.show()

# Adaptive thresholding

# im2 = cv2.adaptiveThreshold(im, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1111, 1)
im3 = cv2.adaptiveThreshold(im, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 1111, 1)
#
# plt.imshow(im2, cmap="gray")
# plt.show()

# plt.imshow(im3, cmap="gray")
# plt.show()

# Straighten the paper

rows, cols, chan = im_src.shape

srcPts = np.float32([[35, 0], [0, rows], [cols, rows-10], [cols-25, 0]])
dstPts = np.float32([[0, 0], [0, rows], [cols-25, rows], [cols, 5]])

M2 = cv2.getPerspectiveTransform(srcPts, dstPts)

print(M2)

im2 = cv2.warpPerspective(im_src, M2, (cols+40, rows+40))

plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.show()


cv2.imwrite('paperAligned.jpg', im2)
