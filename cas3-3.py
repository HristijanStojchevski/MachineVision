import cv2
from matplotlib import pyplot as plt
import numpy as np

im = cv2.imread("c:/Users/petre/PycharmProjects/AllScripts/MV2020/john.jpg",0)
#im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

im = cv2.resize(im, (200, 300), interpolation=cv2.INTER_LINEAR)

plt.imshow(im, cmap='gray')
plt.show()


rows,cols = im.shape

#
#   |1 0 Dx|
#   |0 1 Dy|
#
#M = np.float32([[1,0,50],[0,1,50]])

#M = cv2.getRotationMatrix2D((cols/2,rows/2),80,1.5)


#
#   a = scale * cos(Theta)
#   b = scale * sin(Theta)
#   | a  b  (1-a)*xc-b*yc|
#   | -b a b*xc+(1-a)*yc|
#
#   *
#
#   *    *
#
#        *
#
#   *    *
# srcPts = np.float32([[0,0],[0,10],[10,10]])
# dstPts = np.float32([[0,10],[10,10],[10,0]])
#
# M = cv2.getAffineTransform(dstPts,srcPts)
#
# print(M)
#
#
# im2 = cv2.warpAffine(im,M,(cols+50,rows+50))
#
# srcPts = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])
# dstPts = np.float32([[0,0],[0,rows],[cols+30,rows+30],[cols,0]])
#
#
# M2 = cv2.getPerspectiveTransform(srcPts,dstPts)
#
# print(M2)
#
# im2 = cv2.warpPerspective(im,M2,(cols+100,rows+100))


#im2 = cv2.inRange(im,200,255)

# _,binary = cv2.threshold(im,128,255,cv2.THRESH_BINARY)
# _,binaryinv = cv2.threshold(im,128,255,cv2.THRESH_BINARY_INV)
# tr,binaryotsu = cv2.threshold(im,128,255,cv2.THRESH_OTSU)
# _,binarytozero = cv2.threshold(im,128,255,cv2.THRESH_TOZERO)
# _,binarytozeroinv = cv2.threshold(im,128,255,cv2.THRESH_TOZERO_INV)
# _,binarytrunc = cv2.threshold(im,128,255,cv2.THRESH_TRUNC)


#
#
# print(tr)
#
# # plt.imshow(binary, cmap="gray")
# plt.show()
# plt.imshow(binaryinv, cmap="gray")
# plt.show()
# plt.imshow(binaryotsu, cmap="gray")
# plt.show()
# plt.imshow(binarytozero, cmap="gray")
# plt.show()
# plt.imshow(binarytozeroinv, cmap="gray")
# plt.show()

#plt.imshow(binarytrunc, cmap="gray")
#plt.show()

im2 = cv2.adaptiveThreshold(im,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,111,2)
im3 = cv2.adaptiveThreshold(im,200,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,111,2)




plt.imshow(im2, cmap="gray")
plt.show()


plt.imshow(im3, cmap="gray")
plt.show()
