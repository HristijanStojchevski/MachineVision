import cv2
import matplotlib.pyplot as plt
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


im_src = cv2.imread("../makedonija.png")

im_rgb = cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB)

width, height, channels = im_rgb.shape

im = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)

mask = cv2.inRange(im, 100, 250)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=5)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(mask, contours, -1, 200)
print("Number of contours: ", len(contours))

cnt = contours[0]
# Convex hull
hull = cv2.convexHull(cnt)
cv2.drawContours(mask, [hull], -1, 200)
# Bounding rect
boundingX, boundingY, boundingWidth, boundingHeight = cv2.boundingRect(cnt)
cv2.rectangle(mask, (boundingX, boundingY), (boundingX + boundingWidth, boundingY + boundingHeight), 200)
# Min rectangle
rect = cv2.minAreaRect(cnt)
rectPoints = cv2.boxPoints(rect)
rectPoints = np.int0(rectPoints)
cv2.drawContours(mask, [rectPoints], -1, 128)
# LINE
vx, vy, cx, cy = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
cv2.line(mask, (int(cx - vx * width), int(cy - vy * height)), (int(cx + vx * width), int(cy + vy * height)), 155)

# ELLIPSE
ellipse = cv2.fitEllipse(cnt)
# ellipse = ((278.5, 148.00083923339844), (265, 385), 90.0)
cv2.ellipse(mask, ellipse, 200, 1)

# DEFECTS
hull_def = cv2.convexHull(cnt, returnPoints=False)

defects = cv2.convexityDefects(cnt, hull_def)

for a in range(defects.shape[0]):
    x, y, z, u = defects[a, 0]
    start = tuple(cnt[x][0])
    end = tuple(cnt[y][0])
    far = tuple(cnt[z][0])
    cv2.line(mask, start, end, 50, 4)
    cv2.circle(mask, far, 8, 200, 4)

plt.imshow(mask, cmap='gray')
plt.show()
rand_angle = np.random.uniform(0, 360)
print("Random angle: ", round(rand_angle, 2))
print("Currently with 45 degrees for testing")
mask = cv2.inRange(im, 100, 250)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=5)

m = cv2.getRotationMatrix2D((height/2, width/2), 45, 1.0)
rotated = cv2.warpAffine(mask, m, (height, width))

contours, hierarchy = cv2.findContours(rotated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(len(contours))
cnt = contours[0]

rect = cv2.minAreaRect(cnt)
rectPoints = cv2.boxPoints(rect)
rectPoints = np.int0(rectPoints)
cv2.drawContours(rotated, [rectPoints], -1, 128)

vx, vy, cx, cy = cv2.fitLine(rectPoints, cv2.DIST_HUBER, 0, 0.01, 0.01)
cv2.line(rotated, (int(cx - vx * width), int(cy - vy * width)), (int(cx + vx * width), int(cy + vy * width)), 50)

x_axis = np.array([1, 0])    # unit vector in the same direction as the x axis
line_vector = np.array([vx, vy])  # unit vector in the same direction as your line
dot_product = np.dot(x_axis, line_vector)
angle_np = np.arccos(dot_product)
angle_to_x = angle_between(x_axis, line_vector)

ellipse_rot = cv2.fitEllipse(cnt)
cv2.ellipse(rotated, ellipse_rot, 200, 1)
plt.imshow(rotated, cmap='gray')
plt.show()

print("Angle with ellipse: ", ellipse_rot[2])
print("Angle with line", np.round(angle_to_x[0], 2)*100, 180-angle_np[0]*100)