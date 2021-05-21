import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

sobel_hor = np.ones((3, 3), np.float32)
sobel_ver = np.ones((3, 3), np.float32)
sobel_hor[0, 0] = 1
sobel_hor[0, 1] = 2
sobel_hor[0, 2] = 1
sobel_hor[1, 0] = 0
sobel_hor[1, 1] = 0
sobel_hor[1, 2] = 0
sobel_hor[2, 0] = -1
sobel_hor[2, 1] = -2
sobel_hor[2, 2] = -1
sobel_ver[0, 0] = -1
sobel_ver[0, 1] = 0
sobel_ver[0, 2] = 1
sobel_ver[1, 0] = -2
sobel_ver[1, 1] = 0
sobel_ver[1, 2] = 2
sobel_ver[2, 0] = -1
sobel_ver[2, 1] = 0
sobel_ver[2, 2] = 1


def get_edges(pic):
    sx = cv.filter2D(pic, -1, sobel_hor)
    sy = cv.filter2D(pic, -1, sobel_ver)
    return sx + sy


if __name__ == 'main':
    img = cv.imread('initialPhoto.png', 0)

    img = cv.GaussianBlur(img, (3, 3), 0)
    # sobelx = cv.Sobel(img, -1, 1, 0, ksize=5)  # x
    # sobely = cv.Sobel(img, -1, 0, 1, ksize=5)  # y

    sobelx = cv.filter2D(img, -1, sobel_hor)
    sobely = cv.filter2D(img, -1, sobel_ver)

    sobel = sobelx + sobely
    plt.imshow(sobelx, cmap='gray')
    plt.show()
    plt.imshow(sobely, cmap='gray')
    plt.show()
    plt.imshow(sobel, cmap='gray')
    plt.show()