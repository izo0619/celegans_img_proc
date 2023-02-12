from curses import raw
from tracemalloc import start
import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.io import imread
from conformalmapping import *
from skimage.morphology import (opening, remove_small_objects, area_closing, closing)
from skimage.morphology import disk
from skimage.util.dtype import img_as_uint
from scipy import ndimage
from skimage.measure import label
import numpy as np
from csaps import csaps, CubicSmoothingSpline
import conformalmapping as cm
import scipy
from polygon import polygon
from diskmap import diskmap
import cv2
from aaapy.aaa import aaa
import math
import pandas as pd
from matplotlib.path import Path
import numpy.ma as ma
import standard_worm
from scipy.io import savemat


def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return [original_points[0][indexes[0]], original_points[1][indexes[0]], indexes[0], dist]

def plot_cmplx(z, *a, **k):
    plt.plot(np.real(z), np.imag(z), *a, **k)

# import segmented image
orig_img = imread('./segmentation_correction/h51/h51_sample_seg.tif')
raw_img = imread('./segmentation_correction/h51/h51_sample_seg_orig.tif')
# orig_img = imread('./segmentation_correction/h44/h44_sample_seg.tif')
orig_img = img_as_uint(orig_img)
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
    if value != 0 and value != np.unique(final)[-1]:
        final[final == value] = 0

worm_boundary = segmentation.find_boundaries(final, mode='inner')

pixel_list_row = np.nonzero(worm_boundary)[0]
pixel_list_col = np.nonzero(worm_boundary)[1]

original_points = [pixel_list_col, pixel_list_row]
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

# find center spline and cut into pieces
top_2_curve = np.sort(top_2_curve)
middle_pts_x = [full_bound_data_xi_w[top_2_curve[1]]] # center pts x
middle_pts_y = [full_bound_data_yi_w[top_2_curve[1]]] # center pts y
res = 60 # how many dots to put down each side
spacing_1 = round((top_2_curve[1]-top_2_curve[0])/res)
spacing_2 = round((top_2_curve[0] + (len(full_bound_data_xi) - top_2_curve[1]))/res)
spacing_level = np.linspace(1,res,res)
# piece_widths = [] # width of each piece of worm
# piece_lengths = [] # length of each piece of worm
middle_dist = [0] # array of distances between each middle point
side_1 = [] # array for holding each piece pts on side 1
side_2 = [] # array for holding each piece pts on side 2
pieces = []
for i in range(len(spacing_level)):
    pt1 = (round(top_2_curve[1] - (spacing_1 * spacing_level[i])))
    pt2 = (round(top_2_curve[1] + (spacing_2 * spacing_level[i])) % len(full_bound_data_xi))
    xavg = (full_bound_data_xi_w[pt1] + full_bound_data_xi_w[pt2]) / 2
    yavg = (full_bound_data_yi_w[pt1] + full_bound_data_yi_w[pt2]) / 2
    middle_dist.append(math.dist([middle_pts_x[-1], middle_pts_y[-1]], [xavg, yavg]))
    middle_pts_x.append(xavg)
    middle_pts_y.append(yavg)

    side_1.append(pt1)
    side_2.append(pt2)
    # if this is a selected piece cut, find the width and height
    if ((i+1) % (len(spacing_level)/6) == 0) and i != 0:
        pieces.append(np.concatenate((side_2, side_1[::-1])))
        side_1 = [pt1]
        side_2 = [pt2]
# mapping stage 1: worm to circle
# mapping stage 2: circle to ideal worm
std_worms = standard_worm.gen()
stage_1_maps = []
stage_2_maps = []
stage_1_circles= []
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

# piece = np.empty(np.shape(full_bound_data_xi_w_int[pieces[4]]), dtype=np.complex128); 
# piece.real = full_bound_data_xi_w_int[pieces[4]]
# piece.imag = full_bound_data_yi_w_int[pieces[4]]
# p_spline = Splinep.from_complex_list(piece)
# conformalCenter = np.mean(piece)
# t = np.arange(1000)/1000.
# print(t)
# p_zs_orig = p_spline(t)
# ax3.plot(p_zs_orig.real, p_zs_orig.imag)
# print(p_zs_orig)
# sm = SzMap(p_spline, conformalCenter)
# S = Szego(p_spline, conformalCenter)

# p_zs_circle = np.exp(1.0j * S.theta(t))
# ax3.plot(piece.real, piece.imag)
complex_piece = []
for i in range(len(pieces)):
    pieces[i] = pieces[i][::-1]
    piece = np.empty(np.shape(full_bound_data_xi_w_int[pieces[i]]), dtype=np.complex128); 
    piece.real = full_bound_data_xi_w_int[pieces[i]]
    piece.imag = full_bound_data_yi_w_int[pieces[i]]
    conformalCenter = np.mean(piece)
    # piece = piece - conformalCenter
    complex_piece.append(piece)
    p_spline = Splinep.from_complex_list(piece)
    # std_worm = Splinep.from_complex_list(std_worms[i])
    sm = SzMap(p_spline)
    S = Szego(p_spline)
    t = np.arange(1000)/1000.
    p_zs_orig = p_spline(t)
    p_zs_circle = np.exp(1.0j * S.theta(t))
    ax3.plot(piece.real, piece.imag)
    print("conformal center:")
    print(conformalCenter)
    ax3.scatter([conformalCenter.real], [conformalCenter.imag])
    mat_dict = {'x':piece.real, 'y':piece.imag}
    # ax4.plot(p_zs_circle.real, p_zs_circle.imag)
    # forward map: shape -> circle 
    try:
        r = aaa(p_zs_circle, p_zs_orig, tol=1e-7)
        stage_1_maps.append(r)
        # s = aaa(std_worm, p_zs_circle)
        # stage_2_maps.append(s)
        print("here")
        ax4.plot(r(pieces[i]).real, r(pieces[i]).imag)
    except:
        print(i)
        print('kill myself')
    ax4.legend(["0", "1", "2", "3"])
    # reverse map: circle -> shape 
    # print(std_worm.dtype)
    # s = aaa(std_worm, p_zs_circle)
    # stage_2_maps.append(s)
savemat('wormpiece6_' + str(i) + '.mat', {'cpiece': complex_piece})
# find all points inside worm boundary
x, y = np.meshgrid(np.arange(len(raw_img)), np.arange(len(raw_img))) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x,y)).T 
tup_bounds = tuple(zip(full_bound_data_xi_w, full_bound_data_yi_w))
p = Path(tup_bounds) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(raw_img),len(raw_img)) # now you have a mask with points inside a polygon
raw_img_worm = ma.masked_array(raw_img, mask=np.invert(mask))

# find all points in each piece boundary
raw_img_pieces = []
for i in range(len(pieces)):
    x, y = np.meshgrid(np.arange(len(raw_img)), np.arange(len(raw_img))) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    tup_bounds = tuple(zip(full_bound_data_xi_w_int[pieces[i]], full_bound_data_yi_w_int[pieces[i]]))
    p = Path(tup_bounds) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(len(raw_img),len(raw_img)) # now you have a mask with points inside a polygon
    raw_img_pieces.append(ma.masked_array(raw_img, mask=np.invert(mask)))

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
ax1.imshow(raw_img, cmap=plt.cm.gray)
ax1.axis('off')
ax1.plot(full_bound_data_xi_w, full_bound_data_yi_w, ':o')


ax2.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
ax2.invert_yaxis()
ax2.imshow(raw_img_pieces[0], cmap=plt.cm.gray)
# ax2.plot(full_bound_data_xi_w, full_bound_data_yi_w, '-')
# ax2.plot(middle_pts_x, middle_pts_y, '-')
for i in range(len(pieces)):
    ax2.plot(full_bound_data_xi_w_int[pieces[i]], full_bound_data_yi_w_int[pieces[i]])
    print(full_bound_data_xi_w_int[pieces[i]], full_bound_data_yi_w_int[pieces[i]])


# for i in range(len(stage_1_maps)):
#     # print(stage_1_maps[i](pieces[i]))
#     print(stage_1_maps)
#     ax3.plot(stage_1_maps[i](pieces[i]).real, stage_1_maps[i](pieces[i]).imag, ':o')

plt.show()