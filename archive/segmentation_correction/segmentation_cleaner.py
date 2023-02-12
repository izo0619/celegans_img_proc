import cv2 as cv
import tifffile
import numpy as np
import matplotlib.pyplot as plt
  
# Reading the image from the present directory
image = tifffile.imread("/Users/isabelzhong/Box/10x_wormimages/Results/Full-double-sample-revised-5/h51-well-b-basic-corrected-stiched-nb_Simple-Segmentation.TIF")
image = cv.resize(image, (800, 800))
# image = cv.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX)

kernel = np.ones((5,5), np.uint8)
erosion = cv.erode(image, kernel, iterations=1)
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)


cv.imshow("original", image)
cv.imshow("erosion", erosion)
cv.imshow("opening", opening)
cv.imshow("closing", closing)
cv.waitKey(0)
cv.destroyAllWindows()