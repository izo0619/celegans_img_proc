import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.io import imread
from skimage.morphology import (opening, remove_small_objects, area_closing)
from skimage.morphology import disk
from skimage.util.dtype import img_as_uint
from scipy import ndimage
from skimage.measure import label


# import segmented image
orig_img = imread('./segmentation_correction/h48/h48_sample_seg.tif')
orig_img = img_as_uint(orig_img)

# close any areas smaller than 90 px
area_closed = area_closing(orig_img, 90)
# opening operation
opened = opening(area_closed,disk(7))

# create binary image by setting the background to be black and everything else white
opened[opened == 1] = 1
opened[opened == 2] = 0
opened[opened == 3] = 1
opened[opened == 4] = 1

# fill any holes in worm
fill_worm = ndimage.binary_fill_holes(opened)
# perform opening again to get rid of any salt in the bg
fill_worm = opening(fill_worm, disk(5))
# label the objects in the photo
fill_worm = label(fill_worm)
# remove any object that is less than 20,000 px in area
fill_worm = remove_small_objects(fill_worm, 20000)

raw_img = imread('./segmentation_correction/h48/h48_sample_seg_orig.tif')
raw_img = color.gray2rgb(raw_img)

# 3 image 

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 8), sharex=True,
                                   sharey=True)
ax1.imshow(orig_img, cmap=plt.cm.gray)
ax1.set_title('segmented')
ax1.axis('off')

ax2.imshow(fill_worm.astype('uint8'), cmap=plt.cm.gray)
ax2.set_title('filled worms')
ax2.axis('off')

ax3.imshow(segmentation.mark_boundaries(raw_img, fill_worm), cmap=plt.cm.gray)
ax3.contour(fill_worm, colors='red', linewidths=1)
ax3.set_title('orig')
ax3.axis('off')

plt.show()

# FIRST FEW IMGS 

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16, 8), sharex=True,
#                                    sharey=True)
# ax1.imshow(orig_img, cmap=plt.cm.gray)
# ax1.set_title('segmented')
# ax1.axis('off')

# ax2.imshow(begone_smol, cmap=plt.cm.gray)
# ax2.set_title('begone_smol')
# ax2.axis('off')

# ax3.imshow(opened, cmap=plt.cm.gray)
# ax3.set_title('opening')
# ax3.axis('off')

# ax4.imshow(boundary, cmap=plt.cm.gray)
# ax4.set_title('boundary')
# ax4.axis('off')

# plt.show()


# 6 image
# fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(16, 8), sharex=True,
#                                    sharey=True)
# ax1.imshow(orig_img, cmap=plt.cm.gray)
# ax1.set_title('segmented')
# ax1.axis('off')

# ax2.imshow(opened, cmap=plt.cm.gray)
# ax2.set_title('opening')
# ax2.axis('off')

# ax3.imshow(fill_worm, cmap=plt.cm.gray)
# ax3.set_title('filled')
# ax3.axis('off')

# ax4.imshow(dilated, cmap=plt.cm.gray)
# ax4.set_title('dilation')
# ax4.axis('off')

# ax5.imshow(boundary.astype('uint8'), cmap=plt.cm.gray)
# ax5.set_title('boundary')
# ax5.axis('off')

# ax6.imshow(segmentation.mark_boundaries(raw_img, fill_worm), cmap=plt.cm.gray)
# ax6.contour(fill_worm, colors='red', linewidths=1)
# ax6.set_title('orig')
# ax6.axis('off')

# plt.show()



