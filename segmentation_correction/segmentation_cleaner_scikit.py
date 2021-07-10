import matplotlib.pyplot as plt
from skimage import data
from skimage.util import img_as_ubyte
from skimage.io import imread
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk
from skimage.util.dtype import img_as_uint  # noqa


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

orig_img = imread('./segmentation_correction/sample_seg_small.tif')
# orig_img = orig_img.transpose(1,0,2).reshape(130,-1)
orig_img = img_as_uint(orig_img)
# fig, ax = plt.subplots()
# ax.imshow(orig_img, cmap=plt.cm.gray)
# plt.show()

footprint = disk(6)
eroded = erosion(orig_img, footprint)

# plot_comparison(orig_img, eroded, 'erosion')

dilated = dilation(orig_img,footprint)
# plot_comparison(orig_img, dilated, 'dilation')

opened = opening(orig_img,footprint)
# plot_comparison(orig_img, opened, 'opening')
# open_open = opening(opened, footprint)
# open_erode = erosion(open_open, footprint)
# erode_open_erode = opening(open_erode, footprint)

closed = closing(orig_img,footprint)
# plot_comparison(orig_img, closed, 'closing')

# w_tophat = white_tophat(orig_img, footprint)
# plot_comparison(orig_img, w_tophat, 'white tophat')

# b_tophat = black_tophat(orig_img, footprint)
# plot_comparison(orig_img, b_tophat, 'black tophat')


fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(ncols=5, figsize=(16, 8), sharex=True,
                                   sharey=True)
ax1.imshow(orig_img, cmap=plt.cm.gray)
ax1.set_title('original')
ax1.axis('off')
ax2.imshow(eroded, cmap=plt.cm.gray)
ax2.set_title('erosion')
ax2.axis('off')

# ax2.imshow(opened, cmap=plt.cm.gray)
# ax2.set_title('opening')
# ax2.axis('off')

ax3.imshow(dilated, cmap=plt.cm.gray)
ax3.set_title('dilated')
ax3.axis('off')
# ax3.imshow(opening, cmap=plt.cm.gray)
# ax3.set_title('opening')
# ax3.axis('off')

ax4.imshow(opened, cmap=plt.cm.gray)
ax4.set_title('opening')
ax4.axis('off')
ax5.imshow(closed, cmap=plt.cm.gray)
ax5.set_title('closing')
ax5.axis('off')
plt.show()
