import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors


# PREPROCESSING
df_training = pd.read_csv('data/train.csv')

training_imgs_names = os.listdir('data/Training')
training_imgs = [cv.imread(f'data/Training/{name}') for name in training_imgs_names]
test_imgs_names = os.listdir('data/TestSingleObjects')
test_imgs = [cv.imread(f'data/TestSingleObjects/{name}') for name in test_imgs_names]
dc_training = dict(names=training_imgs_names, imgs=training_imgs)

df_training = pd.DataFrame(dc_training)


# BAG OF WORDS
plt.imshow(df_training.loc[0, 'imgs'])
plt.show()

img = df_training.loc[0, 'imgs']
img1 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

sift = cv.xfeatures2d.SIFT_create()
kp, dsc = sift.detectAndCompute(img1, None)
img = cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img)
plt.show()
