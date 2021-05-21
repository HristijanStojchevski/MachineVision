import cv2
import matplotlib.pyplot as plt

capture = cv2.VideoCapture(0)
plt.ion()
plt.axis('off')
plt.figure(1)
g = cv2.waitKey(1)
while 27 != g:

    _,fr = capture.read()
    # plt.imshow(fr)
    # plt.pause(.1)
    cv2.imshow("Video", fr)
    capturedPhoto = fr
    g = cv2.waitKey(1)

cv2.destroyAllWindows()
capture.release()
# cv2.imshow("Image", capturedPhoto)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img = cv2.resize(capturedPhoto, (600, 400), interpolation=cv2.INTER_CUBIC)
capturedPhoto = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(capturedPhoto)
plt.show()

img2 = cv2.inRange(capturedPhoto, 100, 200)


# RGB
rgbPhoto = capturedPhoto

r = rgbPhoto[:, :, 0] #R
g = rgbPhoto[:, :, 1] #G
b = rgbPhoto[:,:,2] #B

plt.imshow(r, cmap="gray")
plt.show()
plt.imshow(g, cmap="gray")
plt.show()
plt.imshow(b, cmap="gray")
plt.show()
#
# # HSV
# hsvPhoto = cv2.cvtColor(capturedPhoto, cv2.COLOR_RGB2HSV_FULL)
# plt.imshow(hsvPhoto)
# plt.show()
# mask = hsvPhoto[:]
#
# h = hsvPhoto[:,:,0] #R
# s = hsvPhoto[:,:,1] #G
# v = hsvPhoto[:,:,2] #B
#
# img2 = cv2.inRange(h, 50, 250)
# plt.imshow(img2, cmap="gray")
# plt.show()
#
#
# mask[:,:,0]=img2
# mask[:,:,1]=img2
# mask[:,:,2]=img2
#
# mask = cv2.bitwise_not(mask)
# plt.imshow(mask)
# plt.show()
# img_source = cv2.bitwise_and(capturedPhoto, mask)
# plt.imshow(img_source)
# plt.show()
#
# plt.imshow(hsvPhoto)
# plt.show()
# plt.imshow(h, cmap="gray")
# plt.show()
# plt.imshow(s, cmap="gray")
# plt.show()
# plt.imshow(v, cmap="gray")
# plt.show()
#
#
# # LAB
# labPhoto = cv2.cvtColor(capturedPhoto, cv2.COLOR_RGB2LAB)
# plt.imshow(labPhoto)
# plt.show()
