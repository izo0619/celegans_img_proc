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
light = np.copy(opened)
# set boundary and dark to same color (1 - boundary, 2 - bg, 3 - transparent, 4 - dark)
opened[opened == 1] = 4
opened[opened == 3] = 1
filtered_light = np.copy(opened)
# make a copy and dilate it
copy_dilated = np.copy(opened)
copy_dilated = dilation(copy_dilated, disk(7))
filtered_light1 = np.copy(copy_dilated)
# find intersection of dilation and previously light sections
subsetter = np.where((copy_dilated == 4) & (light == 3))
# set intersection as background
opened[subsetter] = 2
# closing operation to fill in holes
opened = closing(opened,disk(3))
# opened[opened == 1] = 3
# opened[opened == 4] = 1
# opened[opened == 3] = 4
# opened = opening(opened,disk(10))
filtered_light2 = np.copy(opened)
# turn into binary image
opened[opened == 2] = 0
opened[opened == 4] = 1
# closing operation again to fill in holes
opened = closing(opened,disk(7))
filtered_light3 = np.copy(opened)

# fill any holes in worm
fill_worm = ndimage.binary_fill_holes(opened)
# perform opening again to get rid of any salt in the bg
fill_worm = opening(fill_worm, disk(5))
# label the objects in the photo
fill_worm = label(fill_worm)
# remove any object that is less than 20,000 px in area
fill_worm = remove_small_objects(fill_worm, 20000)

# only take the last worm
for value in np.unique(fill_worm):
    if value != 0 and value != np.unique(fill_worm)[-1]:
        fill_worm[fill_worm == value] = 0

# find pixel places
# fill_worm.shape => (cols, rows)
worm_boundary = segmentation.find_boundaries(fill_worm, mode='inner')

pixel_list_row = np.nonzero(worm_boundary)[0]
pixel_list_col = np.nonzero(worm_boundary)[1]

def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return [original_points[0][indexes[0]], original_points[1][indexes[0]], indexes[0]]

def find_2_farthest_xy(x_array, y_array, x_point, y_point):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idx1 = np.where(distance==distance.max())
    x_array = np.delete(x_array, idx1)
    y_array = np.delete(y_array, idx1)
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idx2 = np.where(distance==distance.max())
    return [[x_array[idx1[0]][0], x_array[idx2[0]][0]], [y_array[idx1[0]][0], y_array[idx2[0][0]]]]

def minmax(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)

original_points = [pixel_list_col, pixel_list_row]
center = (sum(pixel_list_col) / len(pixel_list_col), sum(pixel_list_row) / len(pixel_list_row))
endpoints = find_2_farthest_xy(pixel_list_col, pixel_list_row, center[0], center[1])
cur_point = [[original_points[0][0], original_points[1][0]]]
cur_idx = 0
sorted_points = [[original_points[0][0]],[original_points[1][0]]]
for i in range(len(pixel_list_col)-1):
    original_points[0] = np.delete(original_points[0], cur_idx)
    original_points[1] = np.delete(original_points[1], cur_idx)
    combined_x_y_arrays = np.dstack([original_points[0].ravel(), original_points[1].ravel()])[0]
    result = do_kdtree(combined_x_y_arrays, cur_point)
    sorted_points[0].append(result[0])
    sorted_points[1].append(result[1])
    cur_point = [result[:-1]]
    cur_idx = result[2]
full_bound_data = csaps(range(len(pixel_list_col)), [sorted_points[0], sorted_points[1]], range(len(pixel_list_col)), smooth=1)
full_bound_data_xi = full_bound_data[0]
full_bound_data_yi = full_bound_data[1]

localization = 10
# x' and y'
x_t = np.gradient(sorted_points[0][::localization])
y_t = np.gradient(sorted_points[1][::localization])
# velocity [x', y']
vel = np.array([ [x_t[i], y_t[i]] for i in range(x_t.size)])
# speed = sqrt(x'^2 + y'^2) = ds/dt
speed = np.sqrt(x_t * x_t + y_t * y_t)
# tangent vector = v/(ds/dt)
tangent = np.array([1/speed] * 2).transpose() * vel

# calculate second deriv
xx_t = np.gradient(x_t)
yy_t = np.gradient(y_t)

# calculate curvature
curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
full_curvature = []
j = 0
for i in range(0,len(pixel_list_col), localization):
    if (i + localization > len(pixel_list_col)):
        full_curvature.extend([curvature_val[j]] * (len(pixel_list_col) - i))
    else:
        full_curvature.extend([curvature_val[j]] * localization)
    j+=1

full_curvature_cumul = []
full_curvature_cumul_val = 0
for i in range(len(curvature_val)):
    full_curvature_cumul_val += curvature_val[i]
    full_curvature_cumul.append(full_curvature_cumul_val)
fcc_x = range(0,len(pixel_list_col), localization)
fcc_s = CubicSmoothingSpline(fcc_x, full_curvature_cumul, smooth=0.00001).spline
fcc_ds1 = fcc_s.derivative(nu=1)
fcc_ds2 = fcc_s.derivative(nu=2)

cross_x = []
cross_y = []
for i in range(len(full_curvature)*100):
    val = fcc_ds2([i*0.01])[0]
    if round(val*10**8) == 0:
        cross_x.append(i*0.01)
        cross_y.append(val)

curve_idx = range(len(full_curvature))
zipped_cur = zip(full_curvature, curve_idx)
zipped_cur = sorted(zipped_cur, reverse=False)
tuples = zip(*zipped_cur)
full_curve_sorted, curve_idx = [ list(tuple) for tuple in  tuples]
portion = round(len(full_curvature)/10)
weighted_indexes = [curve_idx[x:x+portion] for x in range(0, len(curve_idx), portion)]

weights = np.ones_like(sorted_points[0]) * 0.5
w = [0.00000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.1]
for i in range(10):
    weights[weighted_indexes[i]] = w[i]

full_bound_data_w = csaps(range(len(pixel_list_col)), [sorted_points[0], sorted_points[1]], range(len(pixel_list_col)), weights=weights, smooth=0.85)
full_bound_data_xi_w = full_bound_data_w[0]
full_bound_data_yi_w = full_bound_data_w[1]
full_bound_data_xi_w_int = full_bound_data_xi_w.astype(int)
full_bound_data_yi_w_int = full_bound_data_yi_w.astype(int)

raw_img = imread('./segmentation_correction/h44/h44_sample_seg_orig.tif')
raw_img = color.gray2rgb(raw_img)
raw_img[full_bound_data_yi_w_int, full_bound_data_xi_w_int] = (255,0,0)



_, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 6))
ax1.plot(fcc_x, full_curvature_cumul, '.', fcc_x, fcc_s(fcc_x), '-')
ax2.plot(fcc_x, fcc_ds1(fcc_x), '-')
ax3.plot(fcc_x, fcc_ds2(fcc_x), '-')
ax3.plot(cross_x, cross_y, ':o')

ax1.set_title('spline')
ax2.set_title('1st derivative')
ax3.set_title('2nd derivative')
plt.show()

# fig, ax = plt.subplots()
# ax.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
# ax.invert_yaxis()
# # # plt.plot(pixel_list_col, pixel_list_row, 'o')
# plt.plot(sorted_points[0], sorted_points[1], '-', label='sorted points')
# plt.plot(sorted_points[0][1152], sorted_points[1][1152], 'mo', label='start point')
# plt.plot(sorted_points[0][len(sorted_points)-1], sorted_points[1][len(sorted_points)-1], 'mo', label='start point')
# plt.plot(sorted_points[0][385], sorted_points[1][385], 'mo', label='start point')
# # # plt.plot(sorted_points[0][:500], sorted_points[1][:500], 'mo', label='first 500')
# # # plt.plot(full_bound_data_xi, full_bound_data_yi, '-', label='full bound')
# # # plt.scatter(sorted_points[0], sorted_points[1], c=full_curvature, cmap="RdYlGn", s=25)
# # # plt.scatter(sorted_points[0][weighted_indexes[0]], sorted_points[1][weighted_indexes[0]])
# # plt.plot(full_bound_data_xi_w, full_bound_data_yi_w, '-', label='full bound weighted')
# # plt.legend()
# plt.show()

# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.imshow(raw_img, cmap=plt.cm.gray)
# ax1.axis('off')

# ax2.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
# ax2.invert_yaxis()
# plt.plot(sorted_points[0], sorted_points[1], '-', label='sorted points')
# # plt.scatter(sorted_points[0], sorted_points[1], c=full_curvature, cmap="RdYlGn", s=25)
# # plt.plot(full_bound_data_xi_w, full_bound_data_yi_w, '-', label='full bound weighted')
# plt.legend()

# plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)

# ax1.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
# ax1.invert_yaxis()
# ax1.plot(sorted_points[0], sorted_points[1], '-', label='sorted points')
# ax1.legend()

# ax2.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
# ax2.invert_yaxis()
# ax2.scatter(sorted_points[0], sorted_points[1], c=full_curvature, cmap="RdYlGn", s=25, label='curvature')
# ax2.legend()

# ax3.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
# ax3.invert_yaxis()
# ax3.plot(full_bound_data_xi_w, full_bound_data_yi_w, '-', label='full bound weighted')
# ax3.legend()


# plt.show()


# plt.plot(center[0], center[1], ':mo')
# plt.plot(endpoints[0], endpoints[1], ':mo')
# plt.plot(result[0], result[1], 'ro')
# plt.plot(cur_point[0][0], cur_point[0][1], 'bo')
# plt.show()

# hist, bin_edges = np.histogram(full_curvature, 50)
# plt.bar(bin_edges[:-1], hist, width = 1)
# plt.xlim(min(bin_edges), max(bin_edges))
# plt.show()  


# result = find_index_of_nearest_xy(combined_x_y_arrays, cur_point[0][1], cur_point[])
# print(cur_point, result)




# # 3 image 
# # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8), sharex=True,
# #                                    sharey=True)
# fig, (ax1, ax2) = plt.subplots(1,2)
# # ax1.imshow(orig_img, cmap=plt.cm.gray)
# # ax1.set_title('segmented')
# # ax1.axis('off')

# # ax2.imshow(fill_worm.astype('uint8'), cmap=plt.cm.gray)
# # ax2.set_title('filled worms')
# # ax2.axis('off')
# ax1.imshow(segmentation.mark_boundaries(raw_img, fill_worm), cmap=plt.cm.gray)
# ax1.contour(fill_worm, colors='red', linewidths=1)
# ax1.set_title('orig')
# ax1.axis('off')

# ax2.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
# ax2.invert_yaxis()
# ax2.set_title('splined')
# plt.plot(xi, yi, '-', label='center spline (non interpolatable)')
# plt.plot(center['x'], center['y'], 'ro')
# plt.plot(min_points[0][0], min_points[0][1], 'mo', min_points[1][0], min_points[1][1], 'yo')
# plt.plot(half_bound_1_x, half_bound_1_y, '-', label='half bound 1')
# plt.plot(half_bound_2_x, half_bound_2_y, '-', label='half bound 2')
# plt.plot(full_bound_data_xi, full_bound_data_yi, ':', label='full bound')
# plt.legend()

# plt.show()

# 6 image
# fig, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]) = plt.subplots(ncols=3, nrows=3, figsize=(18, 9), sharex=True,
#                                    sharey=True)
# ax1.imshow(orig_img, cmap=plt.cm.gray)
# ax1.set_title('segmented')
# ax1.axis('off')

# ax2.imshow(light, cmap=plt.cm.gray)
# ax2.set_title('opening')
# ax2.axis('off')

# ax3.imshow(filtered_light, cmap=plt.cm.gray)
# ax3.set_title('combine bound and dark')
# ax3.axis('off')

# ax4.imshow(filtered_light1, cmap=plt.cm.gray)
# ax4.set_title('dilate (copy)')
# ax4.axis('off')

# ax5.imshow(filtered_light2, cmap=plt.cm.gray)
# ax5.set_title('set intersection as bg')
# ax5.axis('off')

# ax6.imshow(filtered_light3, cmap=plt.cm.gray)
# ax6.set_title('remove bg')
# ax6.axis('off')

# ax7.imshow(fill_worm, cmap=plt.cm.gray)
# ax7.set_title('filled and cleaned')
# ax7.axis('off')

# ax8.imshow(segmentation.mark_boundaries(raw_img, fill_worm), cmap=plt.cm.gray)
# ax8.contour(fill_worm, colors='red', linewidths=1)
# ax8.set_title('boundary')
# ax8.axis('off')

# ax9.imshow(raw_img, cmap=plt.cm.gray)
# ax9.set_title('spline')
# ax9.axis('off')

# plt.show()


