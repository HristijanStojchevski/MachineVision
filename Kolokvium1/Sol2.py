import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random

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
print(img1.shape)
height, width = img1.shape
img1 = cv.resize(img1, (700, 400), interpolation=cv.INTER_LINEAR)
img = cv.imread('initialPhoto.png')
img = cv.resize(img, (700, 400), interpolation=cv.INTER_CUBIC)
img = cv.GaussianBlur(img, (5, 5), 0)
mask = cv.inRange(img1, 100, 250)
# plt.imshow(mask, cmap='gray')
# plt.show()

white_cells, hier = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
im1 = cv.drawContours(mask, white_cells, -1, (255, 0, 0))
plt.plot()
plt.imshow(im1)
plt.show()

kernel = np.ones((3, 3), np.uint8)
ret, otsu = cv.threshold(img1, 0, 255, cv.THRESH_TRUNC+cv.THRESH_OTSU)
print(ret)

adaptive_gaus = cv.adaptiveThreshold(img1, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# adaptive_gaus_inv = cv.adaptiveThreshold(img1, 150, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 2)
adaptive_mean = cv.adaptiveThreshold(img1, 150, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 2)



_, edges_full = cv.threshold(cv.cvtColor(adaptive_gaus, cv.COLOR_GRAY2RGB), 128, 255, cv.THRESH_BINARY_INV)
_, edges_full_mean = cv.threshold(cv.cvtColor(adaptive_mean, cv.COLOR_GRAY2RGB), 128, 255, cv.THRESH_BINARY_INV)
edges_full_gray = cv.cvtColor(edges_full, cv.COLOR_RGB2GRAY)
edges_full_mean_grey = cv.cvtColor(edges_full_mean, cv.COLOR_RGB2GRAY)


white_edges_canny = cv.Canny(mask, 20, 30)
white_edges_high_tresh = cv.Canny(mask, 60, 120)
white_dilated = sp_noise(white_edges_canny, 0.01)
white_edges_high_tresh = sp_noise(white_edges_high_tresh, 0.01)
white_dilated = cv.dilate(white_dilated, kernel, iterations=3)
white_dilated = cv.erode(white_dilated, kernel, iterations=3)
white_dilated_high = cv.dilate(white_edges_high_tresh, kernel, iterations=3)
white_dilated_high = cv.erode(white_dilated_high, kernel, iterations=3)
white_dilated_high = cv.bilateralFilter(white_dilated_high, 7, 50, 50)
edges_eroded = cv.erode(edges_full_gray, kernel, iterations=1)
edges_eroded_mean = cv.erode(edges_full_mean_grey, kernel, iterations=1)

all_edges_canny = cv.Canny(edges_eroded, 60, 120)
all_edges_canny = sp_noise(all_edges_canny, 0.01)
all_mean_edges = cv.Canny(edges_eroded_mean, 60, 120)
all_mean_edges = sp_noise(all_mean_edges, 0.01)
all_edges_canny = cv.dilate(all_edges_canny, kernel, iterations=1)
all_mean_edges = cv.dilate(all_mean_edges, kernel, iterations=1)
all_edges_canny = cv.erode(all_edges_canny, kernel, iterations=1)
all_mean_edges = cv.erode(all_mean_edges, kernel, iterations=1)


plt.subplot(221)
plt.imshow(white_dilated, cmap='gray')
plt.subplot(222)
plt.imshow(white_dilated_high, cmap='gray')
plt.show()

plt.imshow(all_edges_canny, cmap='gray')
plt.show()

plt.imshow(all_mean_edges, cmap='gray')
plt.show()

plt.imshow(edges_full_gray, cmap='gray')
plt.show()

# contours
all_cells, hierarchy = cv.findContours(edges_full_gray, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
img_cont = cv.drawContours(edges_full_gray, all_cells, -1, (255, 0, 0))

plt.imshow(cv.cvtColor(img_cont, cv.COLOR_GRAY2RGB))
plt.show()
white_cells, hir = cv.findContours(white_dilated_high, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(len(white_cells))
avg_area_white = 0
avg_area_red = 0
i = 0

# threshold ? and cv.contourArea(cnt) > threshold
white_loc = []
maks, mean, avg = 0, 0, 0
minim = len(white_cells[0])
for cnt in white_cells:
    if len(cnt) > maks:
        maks = len(cnt)
    if minim > len(cnt):
        minim = len(cnt)
    avg += len(cnt)
avg = avg/len(white_cells)

for cnt in white_cells:
    (x, y), radius = cv.minEnclosingCircle(cnt)
    if cv.isContourConvex(cnt):
        i += 1
        print("This cell has {} pixels, and location x={}, y={}".format(len(cnt),x,y))
        print("The radius is: {}".format(radius))
        white_loc.append((x, y, radius))
        avg_area_white += len(cnt)
num_whites = i
i = 0
red_loc = []
for cnt in all_cells:
    (x, y), radius = cv.minEnclosingCircle(cnt)
    if cv.isContourConvex(cnt):
        i += 1
        isWhite = False
        for w_x, w_y, w_r in white_loc:
            distance = np.sqrt(np.power((x - w_x), 2) + np.power((y - w_y), 2))
            if (radius + w_r) > distance:
                isWhite = True
        if not isWhite:
            red_loc.append((x, y, radius))
            avg_area_red += len(cnt)
num_reds = i - num_whites
avg_area_white = avg_area_white/num_whites
avg_area_red = avg_area_red/num_reds
rate = num_whites/num_reds  # ratio 1:x

# for loc in white_loc:
#     x = loc[0]
#     y = loc[1]
#     r = loc[2]
#     white_edges_canny = cv.circle(white_edges_canny, (int(x), int(y)), int(r), (255, 0, 0), 5, 8)

plt.imshow(white_edges_canny, cmap='gray')
plt.show()
print("Total whites {}.".format(num_whites))
print("Total reds {}.".format(num_reds))
print("Avg size whites: {}".format(avg_area_white))
print("Avg size reds: {}".format(avg_area_red))
print("Ratio would be 1:{}".format(round(num_reds/num_whites)))

print("White locations list")
print(white_loc)
print("Red location list")
print(red_loc)
