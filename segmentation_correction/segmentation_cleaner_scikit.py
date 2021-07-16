from urllib.request import OpenerDirector
import matplotlib.pyplot as plt
from skimage import data, segmentation, color
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat, remove_small_objects,remove_small_holes, area_closing )
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk
from skimage.util.dtype import img_as_uint  # noqa
import numpy as np
from scipy import ndimage
from skimage.measure import label


def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    plt.show()

def set_diff2d(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)], 'formats':ncols * [A.dtype]}
    C = np.setdiff1d(A.copy().view(dtype), B.copy().view(dtype))
    return C

orig_img = imread('./segmentation_correction/h48/h48_sample_seg.tif')
# orig_img = orig_img.transpose(1,0,2).reshape(130,-1)
orig_img = img_as_uint(orig_img)
# fig, ax = plt.subplots()
# ax.imshow(orig_img, cmap=plt.cm.gray)
# plt.show()

footprint = disk(7)

# eroded = erosion(orig_img, footprint)
# plot_comparison(orig_img, eroded, 'erosion')

# dilated = dilation(orig_img,footprint)
# plot_comparison(orig_img, dilated, 'dilation')
begone_smol = area_closing(orig_img, 30)
opened = opening(begone_smol,footprint)
# plot_comparison(orig_img, opened, 'opening')
# open_open = opening(opened, footprint)
# open_erode = erosion(open_open, footprint)
# erode_open_erode = opening(open_erode, footprint)

# closed = closing(orig_img,footprint)
# plot_comparison(orig_img, closed, 'closing')

# w_tophat = white_tophat(orig_img, footprint)
# plot_comparison(orig_img, w_tophat, 'white tophat')

# b_tophat = black_tophat(orig_img, footprint)
# plot_comparison(orig_img, b_tophat, 'black tophat')

light_layer = np.where(opened == 3)
dark_layer = np.where(opened == 4)
bg_layer = np.where(opened == 2)
bound_layer = np.where(opened == 1)

opened[opened == 1] = 1
opened[opened == 2] = 0
opened[opened == 3] = 0
opened[opened == 4] = 0

opened = closing(opened, disk(10))
# opened = remove_small_objects(opened, 10)
# opened = remove_small_holes(opened, 10)
fill_worm = ndimage.binary_fill_holes(opened)
fill_worm = opening(fill_worm, disk(5))

dilated = dilation(fill_worm, disk(1))

boundary = dilated.copy()
boundary[fill_worm == 1] = 0

raw_img = imread('./segmentation_correction/h48/h48_sample_seg_orig.tif')
raw_img = color.gray2rgb(raw_img)
raw_img[boundary == 1] = (255,0,0)
# raw_img = color.label2rgb(boundary, raw_img, bg_color=(0,0,0))
# print(np.unique(raw_img))
# raw_img[boundary == 1] = 255


# ax2.imshow(eroded, cmap=plt.cm.gray)
# ax2.set_title('erosion')
# ax2.axis('off')
# ax2.imshow(opened, cmap=plt.cm.gray)
# ax2.set_title('opening')
# ax2.axis('off')


# ax3.imshow(opening, cmap=plt.cm.gray)
# ax3.set_title('opening')
# ax3.axis('off')


# ax4.imshow(opened, cmap=plt.cm.gray)
# ax4.set_title('opening')
# ax4.axis('off')

# FIRST FEW IMGS 

# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 8), sharex=True,
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

# ax6.imshow(raw_img, cmap=plt.cm.gray)
# ax6.set_title('orig')
# ax6.axis('off')

# plt.show()



# 3 image 

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 8), sharex=True,
                                   sharey=True)
ax1.imshow(orig_img, cmap=plt.cm.gray)
ax1.set_title('segmented')
ax1.axis('off')

ax2.imshow(boundary.astype('uint8'), cmap=plt.cm.gray)
ax2.set_title('boundary')
ax2.axis('off')

ax3.imshow(raw_img, cmap=plt.cm.gray)
ax3.set_title('orig')
ax3.axis('off')

plt.show()