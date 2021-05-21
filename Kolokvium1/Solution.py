import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from Kolokvium1.Sol1 import get_edges
import random

# function that handles trackbar changes
def doClose(val):
        # create a kernel based on trackbar input
        kernel = np.ones((val,val))
        # do a morphologic close
        res = cv.morphologyEx(img,cv.MORPH_CLOSE, kernel)
        # display result
        cv.imshow("Result", res)


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            #if rdn < prob:
            #    output[i][j] = 0
            if rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


img1 = cv.imread("initialPhoto.png", 0)
# img1 = cv.resize(img1, (700, 400), interpolation=cv.INTER_LINEAR)
img = cv.imread('initialPhoto.png')
img = cv.bilateralFilter(img, 7, 50, 50)
img1 = cv.bilateralFilter(img1, 7, 50, 50)
# img = cv.resize(img, (700, 400), interpolation=cv.INTER_CUBIC)
# capturedPhoto = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img = cv.GaussianBlur(img, (3, 3), 0)
# mask = cv.inRange(img1, 100, 250)

edges = cv.Canny(img, 50, 100)
plt.imshow(edges, cmap='gray')
plt.show()
kernel = np.ones((3, 3), np.uint8)
ret, otsu = cv.threshold(img1, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
print(ret)

adaptive_gaus_inv = cv.adaptiveThreshold(img1, ret, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
adaptive_mean_inv = cv.adaptiveThreshold(img1, ret, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 7, 2)

plt.imshow(adaptive_gaus_inv, cmap='gray')
plt.show()
# plt.imshow(adaptive_mean_inv, cmap='gray')
# plt.show()
edges_full = cv.erode(adaptive_gaus_inv, kernel, iterations=1)
plt.imshow(edges_full, cmap='gray')
plt.show()
# edges_full = cv.bilateralFilter(adaptive_gaus_inv,7,50,50)
edges_full = cv.Canny(edges_full, 50, 100)
edges_full = cv.dilate(edges_full, kernel, iterations=1)
edges_full = cv.erode(edges_full, kernel, iterations=1)
plt.imshow(edges_full, cmap='gray')
plt.show()
#
# # create window and add trackbar
# cv.namedWindow('Result')
# cv.createTrackbar('KernelSize','Result',0,15,doClose)
#
# # display image
# cv.imshow("Result", img)
# cv.waitKey(0)
# cv.destroyAllWindows()
all_cells, hierarchy = cv.findContours(edges_full, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
img_cont = cv.drawContours(edges_full, all_cells, -1, (255, 0, 0))

plt.imshow(cv.cvtColor(img_cont, cv.COLOR_GRAY2RGB))
plt.show()
# white_cells, hir = cv.findContours(white_dilated_high, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# print(len(white_cells))
avg_area_white = 0
avg_area_red = 0
i = 0

# threshold ? and cv.contourArea(cnt) > threshold
# white_loc = []
# maks, mean, avg = 0, 0, 0
# minim = len(white_cells[0])
# for cnt in white_cells:
#     if len(cnt) > maks:
#         maks = len(cnt)
#     if minim > len(cnt):
#         minim = len(cnt)
#     avg += len(cnt)
# avg = avg/len(white_cells)
#
# for cnt in white_cells:
#     (x, y), radius = cv.minEnclosingCircle(cnt)
#     if cv.isContourConvex(cnt):
#         i += 1
#         print("This cell has {} pixels, and location x={}, y={}".format(len(cnt),x,y))
#         print("The radius is: {}".format(radius))
#         white_loc.append((x, y, radius))
#         avg_area_white += len(cnt)
# num_whites = i
# i = 0
red_loc = []
plt.imshow(all_cells, cmap='gray')
plt.show()

for cnt in all_cells:
    (x, y), radius = cv.minEnclosingCircle(cnt)
    red_loc.append((x, y, radius))
    if cv.isContourConvex(cnt):
        i += 1
        red_loc.append((x, y, radius))
        # isWhite = False
        # for w_x, w_y, w_r in white_loc:
        #     distance = np.sqrt(np.power((x - w_x), 2) + np.power((y - w_y), 2))
        #     if (radius + w_r) > distance:
        #         isWhite = True
        # if not isWhite:
        #     red_loc.append((x, y, radius))
        #     avg_area_red += len(cnt)
# num_reds = i - num_whites
# avg_area_white = avg_area_white/num_whites
# avg_area_red = avg_area_red/num_reds
# rate = num_whites/num_reds  # ratio 1:x

# for loc in white_loc:
#     x = loc[0]
#     y = loc[1]
#     r = loc[2]
#     white_edges_canny = cv.circle(white_edges_canny, (int(x), int(y)), int(r), (255, 0, 0), 5, 8)

# plt.imshow(white_edges_canny, cmap='gray')
# plt.show()
# print("Total whites {}.".format(num_whites))
# print("Total reds {}.".format(num_reds))
# print("Avg size whites: {}".format(avg_area_white))
# print("Avg size reds: {}".format(avg_area_red))
# print("Ratio would be 1:{}".format(round(num_reds/num_whites)))
#
# print("White locations list")
# print(white_loc)
print("Red location list")
print(red_loc)