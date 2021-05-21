# LabWork 1
# Author Hristijan Stojchevski
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image

slika = cv2.imread("./Lab1_media/flatwater_kitesurfing.png",1)
# Plot version
slika = cv2.cvtColor(slika,cv2.COLOR_BGR2RGB)

slika = cv2.rectangle(slika,(200,200),(400,400),(255,0,0), 5, 8)
slika = cv2.circle(slika,(450,450),70,(0,255,0),5,8)
slika = cv2.ellipse(slika,(850,720),(60,30),360,0,360,(0,0,255),3,5)

# CV2 version
# slika = cv2.rectangle(slika,(200,200),(400,400),(0,0,255),5,8)
# slika = cv2.circle(slika,(450,450),70,(0,255,0),5,8)
# slika = cv2.ellipse(slika,(850,720),(60,30),360,0,360,(255,0,0),3,5)

cv2.imwrite("Lab1_media/lab1_img.png", slika)
if __name__ == '__main__':
    # plt version
    plt.imshow(slika)
    plt.show()

    # cv2 version
    # cv2.imshow("window",slika)
    # cv2.waitKey(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
