import cv2
import matplotlib.pyplot as plt

capture = cv2.VideoCapture(0)
plt.ion()
plt.axis('off')
plt.figure(1)
g = cv2.waitKey(1)
while 27 != g:

    _, fr = capture.read()
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

# # RGB
# rgbPhoto = capturedPhoto
#
# r = rgbPhoto[:, :, 0]  # R
# g = rgbPhoto[:, :, 1]  # G
# b = rgbPhoto[:, :, 2]  # B
# # img_gr = cv2.inRange(g, 90, 120)
# # plt.imshow(img_gr, cmap='gray')
# # plt.show()
# img_gr = cv2.inRange(g, 90, 130)
# plt.imshow(img_gr, cmap='gray')
# plt.show()
#
# img_r = cv2.inRange(r, 95, 140)
# plt.imshow(img_r, cmap='gray')
# plt.show()
#
#
# img_b = cv2.inRange(b, 50, 100)
# plt.imshow(img_b, cmap='gray')
# plt.show()
# mask = rgbPhoto[:]
# mask[:, :, 0] = img_r
# mask[:, :, 1] = img_gr
# mask[:, :, 2] = img_b
# img_seg = cv2.bitwise_and(img, mask)
# plt.imshow(cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY), cmap='gray')
# plt.show()


# # HSV
# hsvPhoto = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
# plt.imshow(hsvPhoto)
# plt.show()
# mask = hsvPhoto[:]
#
# h = hsvPhoto[:, :, 0]
# s = hsvPhoto[:, :, 1]
# v = hsvPhoto[:, :, 2]
#
# img_h = cv2.inRange(h, 162, 240)
# plt.imshow(img_h, cmap="gray")
# plt.show()
#
# img_s = cv2.inRange(s, 132, 160)
# plt.imshow(img_s, cmap="gray")
# plt.show()
#
# img_v = cv2.inRange(v, 70, 130)
# plt.imshow(img_v, cmap="gray")
# plt.show()
#
# mask[:, :, 0] = img_h
# mask[:, :, 1] = img_s
# mask[:, :, 2] = img_v
#
# # mask_not = cv2.bitwise_not(mask)
# img_seg = cv2.bitwise_and(img, mask)
# plt.imshow(cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY), cmap='gray')
# plt.show()

# # LAB
labPhoto = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
plt.imshow(labPhoto)
plt.show()

mask = labPhoto[:]

l = labPhoto[:, :, 0]
a = labPhoto[:, :, 1]
b = labPhoto[:, :, 2]

img_l = cv2.inRange(l, 73, 115)
plt.imshow(img_l, cmap="gray")
plt.show()

img_a = cv2.inRange(a, 144, 160)
plt.imshow(img_a, cmap="gray")
plt.show()

img_b = cv2.inRange(b, 149, 150)
plt.imshow(img_b, cmap="gray")
plt.show()

mask[:, :, 0] = img_a
mask[:, :, 1] = img_a
mask[:, :, 2] = img_a

img_seg = cv2.bitwise_and(capturedPhoto, mask)
plt.imshow(cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY), cmap='gray')
plt.show()
