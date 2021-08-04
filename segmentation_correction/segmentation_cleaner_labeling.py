import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.io import imread
from skimage.morphology import (opening, remove_small_objects, area_closing)
from skimage.morphology import disk
from skimage.util.dtype import img_as_uint
from scipy import ndimage
from skimage.measure import label, regionprops
import numpy as np
from csaps import csaps
import math

def atan2deg(x):
    x = x if x>=0 else (2*math.pi + x) 
    x*= 180/math.pi
    return x
# import segmented image
orig_img = imread('./segmentation_correction/h21/h21_sample_seg.tif')
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

# only take the last worm
for value in np.unique(fill_worm):
    if value != 0 and value != np.unique(fill_worm)[1]:
        fill_worm[fill_worm == value] = 0

# find pixel places
# fill_worm.shape => (cols, rows)
worm_boundary = segmentation.find_boundaries(fill_worm, mode='outer')

pixel_list_row = np.nonzero(worm_boundary)[0]
pixel_list_col = np.nonzero(worm_boundary)[1]

data_range = range(len(pixel_list_col))
data = [pixel_list_col, pixel_list_row]
# print(len(pixel_list_col))
data_i = csaps(data_range, data, data_range, smooth=0.000001)
xi = data_i[0]
yi = data_i[1]

# find thetas and endpoints
# calculate center
center = {'x': xi[math.ceil(len(xi)/2)], 'y': yi[math.ceil(len(yi)/2)]}
# find two points closest to the spline endpoints
spline_endpoints = [[xi[0], yi[0]], [xi[len(xi)-1], yi[len(yi)-1]]]
min_points = [[center['x'], center['y']], [center['x'], center['y']]]
min_points_dist = [math.dist([center['x'], center['y']], spline_endpoints[0]), 
                    math.dist([center['x'], center['y']], spline_endpoints[1])]
min_points_theta = [0, 0]
thetas = []
for i in range(len(pixel_list_col)):
    angle = math.atan2( (pixel_list_row[i] - center['y']), (pixel_list_col[i] - center['x']))
    thetas.append(atan2deg(angle))
    # check endpoints
    for ii in range(2):
        if math.dist([pixel_list_col[i], pixel_list_row[i]], spline_endpoints[ii]) < min_points_dist[ii]:
            min_points[ii] = [pixel_list_col[i], pixel_list_row[i]]
            min_points_dist[ii] = math.dist([pixel_list_col[i], pixel_list_row[i]], spline_endpoints[ii])
            min_points_theta[ii] = angle
# create tuple triples of angle, x, y and then use that to sort the points onto either side
# sort data based on theta
zipped_lists = zip(thetas, pixel_list_col, pixel_list_row)
zipped_lists = sorted(zipped_lists)
tuples = zip(*zipped_lists)
thetas, pixel_list_col, pixel_list_row = [ list(tuple) for tuple in  tuples]
half_bound_1 = [[],[],[]]
half_bound_2 = [[],[],[]]
sorted_min_points_theta = list(map(atan2deg, min_points_theta))
sorted_min_points_theta.sort()
for tup in zipped_lists:
    # print(atan2deg(tup[0]), atan2deg(min_points_theta[0]))
    if tup[0] > sorted_min_points_theta[0] and tup[0] < sorted_min_points_theta[1]:
        half_bound_1[0].append(tup[0] - sorted_min_points_theta[0])
        half_bound_1[1].append(tup[1])
        half_bound_1[2].append(tup[2])
    elif tup[0] == sorted_min_points_theta[0] or tup[0] == sorted_min_points_theta[1]:
        half_bound_1[0].append(tup[0] - sorted_min_points_theta[0])
        half_bound_1[1].append(tup[1])
        half_bound_1[2].append(tup[2])
        half_bound_2[0].append(tup[0] - sorted_min_points_theta[0]) if tup[0] != sorted_min_points_theta[0] else half_bound_2[0].append(360)
        half_bound_2[1].append(tup[1])
        half_bound_2[2].append(tup[2])
    else:
        if tup[0]-sorted_min_points_theta[0] < 0 :
            half_bound_2[0].append(tup[0] - sorted_min_points_theta[0] + 360)
            half_bound_2[1].append(tup[1])
            half_bound_2[2].append(tup[2])
        else:
            half_bound_2[0].append(tup[0] - sorted_min_points_theta[0])
            half_bound_2[1].append(tup[1])
            half_bound_2[2].append(tup[2])

hb2_zip = zip(half_bound_2[0], half_bound_2[1], half_bound_2[2])
hb2_zip = sorted(hb2_zip)
tuples = zip(*hb2_zip)
half_bound_2[0], half_bound_2[1], half_bound_2[2] = [ list(tuple) for tuple in  tuples]
# half bounds
hb_smoothing = 0.85
# half_bound_1_data = csaps(range(0, len(half_bound_1[1])), [half_bound_1[1], half_bound_1[2]], range(0, len(half_bound_1[1])), smooth=hb_smoothing)
# # half_bound_1_data = csaps(half_bound_1[0], [half_bound_1[1], half_bound_1[2]], range(0, len(half_bound_1[1])), smooth=0.95)
# hb1_xi = half_bound_1_data[0]
# hb1_yi = half_bound_1_data[1]
# # half_bound_2_data = csaps(range(0, len(half_bound_2[1])), [half_bound_2[1], half_bound_2[2]], range(0, len(half_bound_2[1])), smooth=hb_smoothing)
# half_bound_2_data = csaps(half_bound_1[0], [half_bound_1[1], half_bound_1[2]], range(0, len(half_bound_1[1])), smooth=0.95)
# hb2_xi = half_bound_2_data[0]
# hb2_yi = half_bound_2_data[1]
thetas.append(thetas[0] + 360)
pixel_list_row.append(pixel_list_row[0])
pixel_list_col.append(pixel_list_col[0])

half_bound_1_x  = csaps(half_bound_1[0], half_bound_1[1], half_bound_1[0], smooth=hb_smoothing)
half_bound_1_y  = csaps(half_bound_1[0], half_bound_1[2], half_bound_1[0], smooth=hb_smoothing)
half_bound_2_x  = csaps(half_bound_2[0], half_bound_2[1], half_bound_2[0], smooth=hb_smoothing)
half_bound_2_y  = csaps(half_bound_2[0], half_bound_2[2], half_bound_2[0], smooth=hb_smoothing)
full_bound_data = csaps(thetas, [pixel_list_col, pixel_list_row], range(len(pixel_list_col)), smooth=0.85)
full_bound_data_xi = full_bound_data[0]
full_bound_data_yi = full_bound_data[1]

# data = [pixel_list_col, pixel_list_row]
# data_range = range(-math.ceil(len(xi)/2), math.ceil(len(xi)/2))

# spline_obj  = csaps(thetas, data, smooth=0.000001)
# data_i = csaps(thetas, data, data_range, smooth=0.0001)
# xi = data_i[0]
# yi = data_i[1]
# print('data site: ')
# print(0)
# print('x: ')
# print(spline_obj([0])[0])
# print('y ')
# print(spline_obj([0])[1])



# # print(spline_obj.spline)
# fig, ax = plt.subplots()
# ax.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
# ax.invert_yaxis()
# # plt.plot(pixel_list_col, pixel_list_row, 'o')
# plt.plot(xi, yi, '-', label='center spline (non interpolatable)')
# plt.plot(center['x'], center['y'], 'ro')
# plt.plot(min_points[0][0], min_points[0][1], 'mo', min_points[1][0], min_points[1][1], 'yo')
# # plt.plot(half_bound_1[1], half_bound_1[2], ':o')
# # plt.plot(hb1_xi, hb1_yi, '-')
# # plt.plot(half_bound_2[1], half_bound_2[2], ':o')
# # plt.plot(hb2_xi, hb2_yi, '-')
# plt.plot(half_bound_1_x, half_bound_1_y, '-', label='half bound 1')
# plt.plot(half_bound_2_x, half_bound_2_y, '-', label='half bound 2')
# # plt.plot(full_bound_data_xi, full_bound_data_yi, ':', label='full bound')
# # fig, ax = plt.subplots()
# # ax.invert_yaxis()
# # # plt.plot(half_bound_2[0], half_bound_2[1], 'o', label='x values')
# # # plt.plot(half_bound_2[0], half_bound_2[2], 'o', label='y values')
# # plt.plot(thetas, pixel_list_row)
# # plt.plot(thetas, pixel_list_col)
# # ax.set_xlabel('Theta')
# # # # plt.plot(half_bound_1[0], half_bound_1[1], 'o', half_bound_1[0], half_bound_1_x, '-', label='x values')
# # # plt.plot(half_bound_2[0], half_bound_2[1], 'o', half_bound_2[0], half_bound_2_x, '-', label='x values')
# # # plt.plot(half_bound_2[0], half_bound_2[2], 'o', half_bound_2[0], half_bound_2_y, '-', label='y values')
# plt.legend()
# # plt.show()
# # print(thetas)

# x, y = pixel_list_col, pixel_list_row
# xi = np.linspace(x[0], x[-1], 150)
# yi = csaps(x, y, xi, smooth=0.85)

# plt.plot(x, y, 'o', xi, yi, '-')
# plt.show()


raw_img = imread('./segmentation_correction/h21/h21_sample_seg_orig.tif')
raw_img = color.gray2rgb(raw_img)

# 3 image 
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8), sharex=True,
#                                    sharey=True)
fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.imshow(orig_img, cmap=plt.cm.gray)
# ax1.set_title('segmented')
# ax1.axis('off')

# ax2.imshow(fill_worm.astype('uint8'), cmap=plt.cm.gray)
# ax2.set_title('filled worms')
# ax2.axis('off')
ax1.imshow(segmentation.mark_boundaries(raw_img, fill_worm), cmap=plt.cm.gray)
ax1.contour(fill_worm, colors='red', linewidths=1)
ax1.set_title('orig')
ax1.axis('off')

ax2.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
ax2.invert_yaxis()
ax2.set_title('splined')
plt.plot(xi, yi, '-', label='center spline (non interpolatable)')
plt.plot(center['x'], center['y'], 'ro')
plt.plot(min_points[0][0], min_points[0][1], 'mo', min_points[1][0], min_points[1][1], 'yo')
plt.plot(half_bound_1_x, half_bound_1_y, '-', label='half bound 1')
plt.plot(half_bound_2_x, half_bound_2_y, '-', label='half bound 2')
plt.legend()

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



