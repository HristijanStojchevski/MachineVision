import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("./Lab1_media/flatwater_kitesurfing.png")
print(img.shape)
# first part

# plt.imshow(img_source)
# plt.show()
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
#
# # h s v
#
# l = img[:,:,0]
# a = img[:,:,1]
# b = img[:,:,2]
#
# plt.imshow(img_interpolated)
# plt.show()
# plt.imshow(l, cmap="gray")
# plt.show()
# plt.imshow(a, cmap="gray")
# plt.show()
# plt.imshow(b, cmap="gray")
# plt.show()
#
#
# # second part
mask = img[:]
#
h = img[:,:,0] #R
# s = img[:,:,1] #G
# v = img[:,:,2] #B
#
img2 = cv2.inRange(h,50,250)
# plt.imshow(img2,cmap="gray")
# plt.show()
#
#
mask[:,:,0]=img2
mask[:,:,1]=img2
mask[:,:,2]=img2

print(mask)
#
mask = cv2.bitwise_not(mask)
plt.imshow(mask)
plt.show()
img_source = cv2.bitwise_and(img, mask)
plt.imshow(img_source)
plt.show()
#
# plt.imshow(img)
# plt.show()
# plt.imshow(h, cmap="gray")
# plt.show()
# plt.imshow(s, cmap="gray")
# plt.show()
# plt.imshow(v, cmap="gray")
# plt.show()


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]

plt.figure(1)
plt.imshow(img)
plt.show()
plt.imshow(red, cmap="gray")
plt.show()
img2 = red + 50
plt.imshow(img2)
plt.show()
# plt.imshow(green, cmap="gray")
# plt.show()
# plt.imshow(blue, cmap="gray")
# plt.show()

# green = green + 50
#
# img[:,:,1] = green
# plt.imshow(img)
# plt.show()
#
#
# cv2.waitKey()





# capture = cv2.VideoCapture(0)
# plt.ion()
# plt.axis('off')
# plt.figure(1)
# a = cv2.waitKey(1)
# while 27 != a:

#     _,fr = capture.read()
#     plt.imshow(fr)
#     plt.pause(.1)
#     #cv2.imshow("Video",fr)
#     a = cv2.waitKey(1)
#
#
#
# capture.release()