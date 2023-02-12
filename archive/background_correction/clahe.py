import cv2
import tifffile
import numpy as np
  
# Reading the image from the present directory
image = tifffile.imread("/Users/isabelzhong/Box/10x_wormimages/p21-growth-H21-10X_Plate_3233/stiched_images/well-a-stiched.TIF")
# image = cv2.imread("/Users/isabelzhong/Downloads/flappybird.png")

# Resizing the image for compatibility
# image = cv2.resize(image, (800, 800))
image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
   
# The initial processing of the image
# image = cv2.medianBlur(image, 3)
# image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# The declaration of CLAHE 
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit=15, tileGridSize=(64,64))
final_img = clahe.apply(image)
  
# Ordinary thresholding the same image
# _, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
  
# Showing all the three images
# cv2.imshow("ordinary threshold", ordinary_img)
cv2.imshow("original", image)
cv2.imshow("CLAHE image", final_img)
# cv2.imwrite("clahe-clip_15-tilesize_64.TIF", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# # img = cv.imread('/Users/isabelzhong/Downloads/flappybird.png',0)
# img = cv.imread("/Users/isabelzhong/Box/10x_wormimages/p21-growth-H21-10X_Plate_3233/stiched_images/well-a-stiched.TIF")
# # create a CLAHE object (Arguments are optional).
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv.imshow('clahe_2.jpg',cl1)
# cv.waitKey(0)
# cv.destroyAllWindows()
