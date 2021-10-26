import cv2
import numpy as np
import matplotlib.pyplot as plt

# template = cv2.imread('../template.png', 0)
#
# img_src = cv2.imread('../lena.png', 1)
# img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
#
# plt.imshow(img_gray, cmap='gray')
# plt.show()
#
# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
# names = ['TM_CCOEF', 'TM_SQDIFF', 'TM_SQDIFF_NORM', 'TM_CCOEF_NORMED']
# print(img_gray)
# for method, name in zip(methods, names):
#     temp = np.empty_like(img_gray)
#     np.copyto(temp, img_gray)
#     res = cv2.matchTemplate(temp, template, method)
#     # plt.imshow(res, cmap='gray')
#     # plt.show()
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     print(min_loc, max_loc)
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         topleft = min_loc
#     else:
#         topleft = max_loc
#     bottomright = (topleft[0]+50, topleft[1]+50)
#     cv2.rectangle(temp, topleft, bottomright, 255, 4)
#     plt.title(name)
#     plt.imshow(temp, cmap='gray')
#     plt.show()


#  FLAG LINES AND CIRCLE
img_src = cv2.imread('../makedonija.png', 1)
img_bgr = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
# plt.imshow(thresh, cmap='gray')
# plt.show()
#
edges = cv2.Canny(img_gray, 30, 180)
# # lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
# plt.imshow(edges, cmap='gray')
# plt.show()
# # lines2 = cv2.HoughLinesP(edges, 1.5, np.pi/180, 96, minLineLength=30, maxLineGap=30)
# # # lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 95, minLineLength=0, maxLineGap=30)
# #
# circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0, maxRadius=0)
#
# circles = np.uint16(np.around(circles))
# for c in circles[0, :]:
#     cv2.circle(img_bgr, (c[0], c[1]), c[2], (0, 255, 0), 2)
#     cv2.circle(img_bgr, (c[0], c[1]), 2, (255, 255, 255), 2)


# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0+1000*(-b))
#     y1 = int(y0+1000*a)
#     x2 = int(x0-1000*(-b))
#     y2 = int(y0-1000*a)
# for line in lines2:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#
# cv2.drawContours(img_bgr, contours, -1, (0, 255, 0))
#
# plt.imshow(img_bgr)
# plt.show()

# FLAG WATERSHED
# cv2.drawContours(img_gray, contours, -1, 0)
# ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

plt.imshow(thresh, cmap='gray')
plt.show()
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
plt.imshow(opening, cmap='gray')
plt.show()

sure_bg = cv2.dilate(thresh, kernel, iterations=4)

# dist_tr = cv2.distanceTransform(opening, cv2.DIST_C, 0)
dist_tr = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

ret2, sure_fg = cv2.threshold(dist_tr, 0.5*dist_tr.max(), 255, 0)

plt.imshow(dist_tr, cmap='gray')
plt.show()

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

plt.title('Sure background')
plt.imshow(sure_bg, cmap='gray')
plt.show()
plt.title('Sure foreground')
plt.imshow(sure_fg, cmap='gray')
plt.show()
plt.title('UNKNOWN')
plt.imshow(unknown, cmap='gray')
plt.show()

ret, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1

markers[unknown == 255] = 0

plt.imshow(markers)
plt.show()

markers = cv2.watershed(img_bgr, markers)
img_bgr[markers == -1] = [0, 255, 0]
plt.imshow(img_bgr)
plt.show()
plt.imshow(markers)
plt.show()