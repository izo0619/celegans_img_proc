import matlab.engine
import matlab
import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.io import imread
from skimage.morphology import (opening, remove_small_objects, area_closing, closing)
from skimage.morphology import disk
from skimage.util.dtype import img_as_uint
from scipy import ndimage
from skimage.measure import label
import numpy as np
from csaps import csaps, CubicSmoothingSpline
import scipy
from matplotlib.path import Path
from scipy.interpolate import griddata
from stripmap.map import Stripmap, Polygon

def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return [original_points[0][indexes[0]], original_points[1][indexes[0]], indexes[0], dist]

def plot_cmplx(z, *a, **k):
    plt.plot(np.real(z), np.imag(z), *a, **k)

# import segmented image
orig_img = imread('../segmentation_correction/h51/h51_sample_seg.tif')
raw_img = imread('../segmentation_correction/h51/h51_sample_seg_orig.tif')
# orig_img = imread('./segmentation_correction/h44/h44_sample_seg.tif')
orig_img = img_as_uint(orig_img)
# print(orig_img)

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
print(np.unique(final))
unique_worms = np.unique(final)
for value in unique_worms:
    if value != 0 and value != unique_worms[1]:
        final[final == value] = 0
# plt.imshow(final)
# plt.show()
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

# find 48 points around worm
top_2_curve = np.sort(top_2_curve)
conf_map_points = [top_2_curve[0]]
spacing_1 = round((top_2_curve[1]-top_2_curve[0])/36)
spacing_2 = round((top_2_curve[0] + (len(full_bound_data_xi) - top_2_curve[1]))/36)
# spacing_level = [0.5,2,3,8,12,16,20,24,28,33,34,35.5]
spacing_level = [0.5,1,2,3,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,33,34,35,35.5]
for i in range(len(spacing_level)):
    conf_map_points.append(round(top_2_curve[0] + (spacing_1 * spacing_level[i])))
conf_map_points.append(top_2_curve[1])
for i in range(len(spacing_level)):
    conf_map_points.append(round(top_2_curve[1] + (spacing_2 * spacing_level[i])) % len(full_bound_data_xi))

# create find interior points
x, y = np.meshgrid(np.arange(len(raw_img[0])), np.arange(len(raw_img))) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x,y)).T 
# tup_bounds = tuple(zip(full_bound_data_xi_w_int, full_bound_data_yi_w_int))
tup_bounds = tuple(zip(full_bound_data_xi_w[conf_map_points], full_bound_data_yi_w[conf_map_points]))
p = Path(tup_bounds) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(raw_img[0]),len(raw_img)) # mask with points inside a polygon
mask_points = np.argwhere(mask)
interior_points = mask_points[:,1] + 1j*mask_points[:,0]
interior_points_idx = np.argwhere(grid)
colors = raw_img.flatten()[interior_points_idx].flatten()
# plt.scatter(interior_points.real, interior_points.imag)
# plt.show()

# feed into matlab ** MUST BE COUNTERCLOCKWISE DIRECTION **
## reverse pts before feed
full_worm_ccw_ext = np.array([full_bound_data_xi_w[conf_map_points][::-1], full_bound_data_yi_w[conf_map_points][::-1]])
# full_worm_ccw_ext = np.array([full_bound_data_xi_w[conf_map_points], full_bound_data_yi_w[conf_map_points]])
full_worm_ends_ccw = [1,24]
# plt.scatter(full_worm_ccw_ext[0], full_worm_ccw_ext[1], c=range(len(full_worm_ccw_ext[0])), cmap="cool")
# plt.scatter(full_worm_ccw_ext[0][full_worm_ends_ccw[0]], full_worm_ccw_ext[1][full_worm_ends_ccw[0]], color = 'red')
# plt.scatter(full_worm_ccw_ext[0][full_worm_ends_ccw[1]], full_worm_ccw_ext[1][full_worm_ends_ccw[1]], color = 'red')
# plt.show()
p = Polygon(full_worm_ccw_ext[0].tolist(), full_worm_ccw_ext[1].tolist())
stripmap = Stripmap(p, list(full_worm_ends_ccw))
print("map created")
res = 1000
x_interior = mask_points[:,1].tolist()[::res]
y_interior = mask_points[:,0].tolist()[::res]
interior_mapped = stripmap.evalinv(x_interior, y_interior)

fig, axs = plt.subplots(2, 1)
axs[0].plot(full_worm_ccw_ext[0].tolist(), full_worm_ccw_ext[1].tolist())
axs[0].set_title("original")
axs[1].scatter(interior_mapped[0], interior_mapped[1], c=colors[::res], cmap='gray')
axs[1].set_title("inverse mapped")
fig.tight_layout()
plt.show()

# set up output to become list of lists
result = [list(i) for i in zip(interior_mapped[0], interior_mapped[1])]
#generate grid data using mgrid
grid_x,grid_y = np.mgrid[0:25:25000j, 0:1:1000j]
# grid_a = griddata(result, colors, (grid_x, grid_y), method='cubic')
grid_b = griddata(result, colors[::res], (grid_x, grid_y), method='linear')
grid_c = griddata(result, colors[::res], (grid_x, grid_y), method='nearest')
# print(grid_a)
# print(grid_b)
# print(grid_c)
# plt.imshow(grid_c.T, cmap='gray')
# plt.show()

fig, axs = plt.subplots(2, 1)
axs[1].imshow(grid_b.T, cmap='gray')
axs[1].set_title("linear")
axs[0].imshow(grid_c.T, cmap='gray')
axs[0].set_title("nearest")
fig.tight_layout()
plt.show()
print("complete")