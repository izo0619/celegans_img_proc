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
orig_img = imread('./segmentation_correction/h51/h51_sample_seg.tif')
# orig_img = imread('./segmentation_correction/h44/h44_sample_seg.tif')
orig_img = img_as_uint(orig_img)

# idxB = np.where(orig_img == 1)
# idxBG = np.where(orig_img == 2)
# idxT = np.where(orig_img == 3)
# idxD = np.where(orig_img == 4)

# combined = np.zeros_like(orig_img)
# combined[idxB] = 1
# combined[idxD] = 1

# se = disk(20)
# filled = ndimage.binary_fill_holes(combined)
# filled = label(filled)
# filled = remove_small_objects(filled, 20000)
# final = closing(filled, se)

# close any areas smaller than 90 px
area_closed = area_closing(orig_img, 90)
# opening operation
opened = opening(area_closed,disk(5))
state0 = np.copy(opened)

opened[opened == 1] = 1
opened[opened == 2] = 0
opened[opened == 3] = 0
opened[opened == 4] = 1

# perform opening again to get rid of any salt in the bg
# fill_worm = opening(opened,disk(1))
fill_worm = label(opened)
fill_worm = remove_small_objects(fill_worm, 1000)
state1 = np.copy(fill_worm)
fill_worm = closing(fill_worm, disk(15))
state2 = np.copy(fill_worm)
# fill any holes in worm
fill_worm = ndimage.binary_fill_holes(fill_worm)
state3 = np.copy(fill_worm)
# label the objects in the photo
fill_worm = label(fill_worm)
state4 = np.copy(fill_worm)
# remove any object that is less than 20,000 px in area
final = remove_small_objects(fill_worm, 20000)
state5 = np.copy(final)

# only take the last worm
for value in np.unique(final):
    if value != 0 and value != np.unique(final)[1]:
        final[final == value] = 0

worm_boundary = segmentation.find_boundaries(final, mode='inner')

pixel_list_row = np.nonzero(worm_boundary)[0]
pixel_list_col = np.nonzero(worm_boundary)[1]

def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return [original_points[0][indexes[0]], original_points[1][indexes[0]], indexes[0], dist]

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
# endpoints = find_2_farthest_xy(pixel_list_col, pixel_list_row, center[0], center[1])
cur_point = [[original_points[0][0], original_points[1][0]]]
cur_idx = 0
sorted_points = [[original_points[0][0]],[original_points[1][0]]]
tot_dist = 0
for i in range(len(pixel_list_col)-1):
    original_points[0] = np.delete(original_points[0], cur_idx)
    original_points[1] = np.delete(original_points[1], cur_idx)
    combined_x_y_arrays = np.dstack([original_points[0].ravel(), original_points[1].ravel()])[0]
    result = do_kdtree(combined_x_y_arrays, cur_point)
    tot_dist += result[3]
    if result[3] > ((tot_dist/(i+1))*100): break
    sorted_points[0].append(result[0])
    sorted_points[1].append(result[1])
    cur_point = [result[:2]]
    cur_idx = result[2]
full_bound_data = csaps(range(len(sorted_points[0])), [sorted_points[0], sorted_points[1]], range(len(sorted_points[0])), smooth=1)
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
for i in range(0,len(full_bound_data_xi), localization):
    if (i + localization > len(full_bound_data_xi)):
        full_curvature.extend([curvature_val[j]] * (len(full_bound_data_xi) - i))
    else:
        full_curvature.extend([curvature_val[j]] * localization)
    j+=1

# calculate cumulative curvature and find points of local maximum curvature
full_curvature_cumul = []
full_curvature_cumul_val = 0
for i in range(len(curvature_val)):
    full_curvature_cumul_val += curvature_val[i]
    full_curvature_cumul.append(full_curvature_cumul_val)
fcc_x = range(0,len(full_bound_data_xi), localization)
fcc_s = CubicSmoothingSpline(fcc_x, full_curvature_cumul, smooth=0.00001).spline
fcc_ds1 = fcc_s.derivative(nu=1)
fcc_ds2 = fcc_s.derivative(nu=2)
top_2_curve = (fcc_ds2.roots()[(fcc_ds1(fcc_ds2.roots())).argsort()])[-2:].astype(int)

# sort curvature and split weights
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

full_bound_data_w = csaps(range(len(full_bound_data_xi)), [sorted_points[0], sorted_points[1]], range(len(full_bound_data_xi)), weights=weights, smooth=0.85)
full_bound_data_s = CubicSmoothingSpline(range(len(full_bound_data_xi)), [sorted_points[0], sorted_points[1]], weights=weights, smooth=0.85).spline
full_bound_data_xi_w = full_bound_data_w[0]
full_bound_data_yi_w = full_bound_data_w[1]
full_bound_data_xi_w_int = full_bound_data_xi_w.astype(int)
full_bound_data_yi_w_int = full_bound_data_yi_w.astype(int)

# find 20 points around worm
conf_map_points = [top_2_curve[0]]
spacing = round(abs(top_2_curve[0]-top_2_curve[1])/18)
spacing_level = [1,2,3,6,9,12,15,16,17]
for i in range(9):
    conf_map_points.append(round(top_2_curve[0] + (spacing * spacing_level[i])) % len(full_bound_data_xi))
conf_map_points.append(top_2_curve[1])
for i in range(9):
    conf_map_points.append(round(top_2_curve[1] + (spacing * spacing_level[i])) % len(full_bound_data_xi))


raw_img = imread('./segmentation_correction/h51/h51_sample_seg_orig.tif')
raw_img = color.gray2rgb(raw_img)
raw_img[sorted_points[1], sorted_points[0]] = (255,0,0)


# _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 6))
# ax1.plot(fcc_x, full_curvature_cumul, '.', fcc_x, fcc_s(fcc_x), '-', label='cumulative curvature')
# ax2.plot(fcc_x, fcc_ds1(fcc_x), '-')
# ax2.plot(top_2_curve, fcc_ds1(top_2_curve), 'o', label='top 2 local max')
# ax3.plot(fcc_x, fcc_ds2(fcc_x), '-')
# ax3.plot(fcc_ds2.roots(), fcc_ds2(fcc_ds2.roots()), ':o', label='roots')

# ax1.set_title('spline')
# ax1.set_ylabel('curvature')
# ax1.legend()
# ax2.set_title('1st derivative')
# ax2.legend()
# ax3.set_title('2nd derivative')
# ax3.set_xlabel('point idx')
# ax3.legend()
# plt.show()

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(raw_img, cmap=plt.cm.gray)
ax1.axis('off')
ax1.plot(full_bound_data_xi_w[conf_map_points], full_bound_data_yi_w[conf_map_points], ':o')

# ax2.imshow(final, cmap=plt.cm.gray)
# ax2.axis('off')
ax2.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
ax2.invert_yaxis()
ax2.plot(sorted_points[0], sorted_points[1], '-', label='sorted points')
ax2.plot(full_bound_data_xi_w[conf_map_points], full_bound_data_yi_w[conf_map_points], ':o')
ax2.legend()
plt.show()

# fig, ([ax1, ax2, ax3], [ax4, ax5, ax6]) = plt.subplots(ncols=3, nrows=2, figsize=(18, 9), sharex=True,
#                                    sharey=True)
# ax1.imshow(state0, cmap=plt.cm.gray)
# ax1.set_title('segmented')
# ax1.axis('off')

# ax2.imshow(state1, cmap=plt.cm.gray)
# ax2.set_title('opening')
# ax2.axis('off')

# ax3.imshow(state2, cmap=plt.cm.gray)
# ax3.set_title('combine bound and dark')
# ax3.axis('off')

# ax4.imshow(state3, cmap=plt.cm.gray)
# ax4.set_title('dilate (copy)')
# ax4.axis('off')

# ax5.imshow(state4, cmap=plt.cm.gray)
# ax5.set_title('set intersection as bg')
# ax5.axis('off')

# ax6.imshow(state5, cmap=plt.cm.gray)
# ax6.set_title('remove bg')
# ax6.axis('off')

# plt.show()