import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.io import imread
from skimage.morphology import (opening, remove_small_objects, area_closing, closing)
from skimage.morphology import disk
from skimage.morphology.grey import dilation, erosion
from skimage.util.dtype import img_as_uint
from scipy import ndimage
from skimage.measure import label, regionprops
import numpy as np
from csaps import csaps, CubicSmoothingSpline
import math
import scipy
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def atan2deg(x):
    x = x if x>=0 else (2*math.pi + x) 
    x*= 180/math.pi
    return x
# import segmented image
orig_img = imread('./segmentation_correction/h44/h44_sample_seg.tif')
# orig_img = imread('./segmentation_correction/h44/h44_sample_seg.tif')
orig_img = img_as_uint(orig_img)

# close any areas smaller than 90 px
area_closed = area_closing(orig_img, 90)
# opening operation
opened = opening(area_closed,disk(7))

opened[opened == 1] = 1
opened[opened == 2] = 0
opened[opened == 3] = 0
opened[opened == 4] = 1

# perform opening again to get rid of any salt in the bg
fill_worm = closing(opened, disk(15))
# fill any holes in worm
fill_worm = ndimage.binary_fill_holes(fill_worm)
# label the objects in the photo
fill_worm = label(fill_worm)
# remove any object that is less than 20,000 px in area
final = remove_small_objects(fill_worm, 20000)

# only take the last worm
for value in np.unique(final):
    if value != 0 and value != np.unique(final)[1]:
        final[final == value] = 0

worm_boundary = segmentation.find_boundaries(final, mode='inner')

pixel_list_row = np.nonzero(worm_boundary)[0]
pixel_list_col = np.nonzero(worm_boundary)[1]


# rng = np.random.default_rng()
# points = rng.random((30, 2))
points = np.dstack([pixel_list_row.ravel(), pixel_list_col.ravel()])[0]
# print(points)
hull = ConvexHull(points)
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')
plt.show()
