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
import conformalmapping as cm
import scipy
from polygon import polygon
from diskmap import diskmap
import cv2
from aaapy.aaa import aaa

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

# find 20 points around worm
top_2_curve = np.sort(top_2_curve)
conf_map_points = [top_2_curve[0]]
spacing_1 = round((top_2_curve[1]-top_2_curve[0])/36)
spacing_2 = round((top_2_curve[0] + (len(full_bound_data_xi) - top_2_curve[1]))/36)
# spacing_level = [1,2,3,4,6,8,10,12,14,15,16,17]
spacing_level = [0.5,2,3,8,12,16,20,24,28,33,34,35.5]
for i in range(len(spacing_level)):
    conf_map_points.append(round(top_2_curve[0] + (spacing_1 * spacing_level[i])))
conf_map_points.append(top_2_curve[1])
for i in range(len(spacing_level)):
    conf_map_points.append(round(top_2_curve[1] + (spacing_2 * spacing_level[i])) % len(full_bound_data_xi))
# conf_map_points.append(top_2_curve[0])


raw_img = color.gray2rgb(raw_img)
raw_img[sorted_points[1], sorted_points[0]] = (255,0,0)

#### original dir ########
conf_map_points = np.sort(conf_map_points)
# # create spline obj for confmap
G = cm.Splinep(full_bound_data_xi_w[conf_map_points],full_bound_data_yi_w[conf_map_points])

# #### reverse dir #######
# conf_map_points = [ len(full_bound_data_xi)- 1 - x for x in conf_map_points ]
# conf_map_points = np.sort(conf_map_points)
# # # create spline obj for confmap
# G = cm.Splinep(full_bound_data_xi_w[::-1][conf_map_points],full_bound_data_yi_w[::-1][conf_map_points])


# creates riemann map via szego kernel, not sure what happens here but it just looks like the normal boundary when plotted??
sm = cm.SzMap(G, 0)
# szego object, identified by curve, center, and a bunch of kernel properties
S = cm.Szego(G, 0)
# points along the boundary, this is defined by some ratio of 0-1 along the spline
# t =  [ x / len(full_bound_data_xi)for x in conf_map_points]
t = np.arange(3000)/3000. #create 3000 points between 0 and 1

# domain (worm)
zs = G(t) # G is the spline, so grab the t points around the worm
# range (circle)
zs_2 = np.exp(1.0j * S.theta(t)) # their corresponding map on the unit circle

# np.set_printoptions(precision=4, suppress=True, linewidth=15)
# N = 2
# th= 2*np.pi*np.arange(N)/float(N)
# t_2 = S.invtheta(th)
# w = G(t_2)
# c = np.fft.fft(w)/float(N)

s = aaa(zs, zs_2)

# f = lambda z : np.polyval(cm.helpers.flipud(c),z)
gd = cm.unitdisk().grid() # set of inner unit circle curves
lst = []
for curve in gd.curves:
    newcurve = s(curve) # for each curve, find the corresponding worm curve
    # newcurve = sm.applyMap(curve)
    lst.append(newcurve)
# print(lst)
gc = cm.GridCurves(lst)
# gc.plot()
orig_gc = cm.GridCurves(gd.curves)
# orig_gc.plot()
# G.plot()
# plt.gca().set_aspect('equal')
# plt.gca().axis(G.plotbox())
# ax = plt.gca()
# ax.set_xticks([]) 
# ax.set_yticks([]) 
# plt.show()

plt.subplot(1,2,1)
gc.plot()
G.plot()
zs = G(t)
zs_2 = np.exp(1.0j * S.theta(t))
# plt.gca().invert_yaxis()
# plt.scatter(w.real, w.imag, c=t_2, cmap="cool", s=30)
# plt.plot(sm.applyMap([0.1 + 0.5j]).real, sm.applyMap([0.1 + 0.5j]).imag, 'bo')
# plt.plot(sm.applyMap([0.3+0.2j]).real, sm.applyMap([0.3+0.2j]).imag, 'mo')
# plt.plot(sm.applyMap([0 + 1j]).real, sm.applyMap([0 + 1j]).imag, 'go')
# plt.plot(sm.applyMap([1 + 0j]).real, sm.applyMap([1 + 0j]).imag, 'go')
# plt.plot(sm.applyMap([0 - 1j]).real, sm.applyMap([0 - 1j]).imag, 'go')
# plt.plot(s([-1 + 0j, 0 - 1j, 1 + 0j, 0 + 1j]).real, s([-1 + 0j, 0 - 1j, 1 + 0j, 0 + 1j]).imag, 'go')
# plt.plot(s([zs_2[20]]).real, s([zs_2[20]]).imag, 'go')
plt.scatter(s(zs_2).real, s(zs_2).imag, c=t, cmap="cool", s=30)
# plt.plot(sm.applyMap([1+0j]).real, sm.applyMap([1+0j]).imag, 'ro')
# plt.plot(sm.applyMap(zs[0]).real, sm.applyMap(zs[0]).imag, 'yo')
# plt.plot(sm.applyMap([0+0j]).real, sm.applyMap([0+0j]).imag, 'co')
# plt.plot(com[0], com[1], 'ro')

plt.subplot(1,2,2)
c = cm.Circle(0, 1)
c.plot()
# calculates where each point goes in the circle???
# zs_2 = np.exp(1.0j * S.theta(t))
# plt.plot(zs.real[1], zs.imag[1], 'ro', fillstyle='none')
# plt.plot(zs.real[0], zs.imag[0], 'ro', fillstyle='none')
# plt.scatter(zs_2.real, zs_2.imag, c=t, cmap="cool", s=30)
orig_gc.plot()
# plt.plot(0.1, 0.5, 'bo')
# plt.plot(0.3, 0.2, 'mo')
# plt.plot(0, 1, 'go')
# plt.plot(1, 0, 'go')
# plt.plot(0, -1, 'go')
# plt.plot(-1, 0, 'go')
# plt.plot(zs_2[20].real, zs_2[20].imag, 'go')
plt.scatter(zs_2.real, zs_2.imag, c=t, cmap="cool", s=30)
# plt.plot(1, 0, 'ro')
# plt.plot(zs.real[0], zs.imag[0], 'yo')
# plt.plot(0, 0, 'co')
plt.gca().set_aspect('equal')
plt.gca().axis(c.plotbox())
plt.show()

# pts2 = np.float32(np.dstack([(zs.real[:3]*np.shape(orig_img)[0]).ravel(), (zs.imag[:3]*np.shape(orig_img)[1]).ravel()])[0])
# print(pts2)
# M = cv2.getAffineTransform(pts1, pts2)
# dst = cv2.warpAffine(orig_img,M,np.shape(orig_img))
# plt.subplot(121)
# plt.imshow(orig_img, cmap=plt.cm.gray)
# plt.title('Input')
  
# plt.subplot(122)
# plt.imshow(dst, cmap=plt.cm.gray)
# plt.title('Output')
  
# plt.show()

# G = np.dstack([full_bound_data_xi_w[conf_map_points].ravel(), full_bound_data_yi_w[conf_map_points].ravel()])[0]
# G = np.array(list(map(lambda c: np.complex(*c), G)))
# map_1 = np.float32(np.dstack([zs.real.ravel(), zs.imag.ravel()])[0])
# print(map_1)
# map_2 = np.float32(np.dstack([zs.real.ravel(), zs.imag.ravel()])[0])
# # M = cv2.getPerspectiveTransform(map_1,map_2)
# # dst = cv2.warpPerspective(orig_img,M,np.shape(orig_img))
# dst = cv2.remap(raw_img, map_1, map_2, cv2.INTER_LINEAR)

# plt.subplot(121)
# plt.imshow(raw_img)
# plt.title('image')
# plt.subplot(122)
# plt.imshow(dst)
# plt.title('remapped')
# plt.show()
# plt.plot(zs.real, zs.imag, 'ro')
# plt.gca().set_aspect('equal')
# plt.gca().axis(G.plotbox())

# p = polygon(G)
# m = diskmap(p)

# x,y = np.meshgrid(np.linspace(-0.8, 0.8, 9),
#                   np.linspace(-0.99, 0.99, 100))

# z = (x + y*1j)*(0.5 + 0.5j)
# zi = z*1j
# theta = np.linspace(0, 2*np.pi, 100)

# plt.figure(figsize=(6.4, 3.2))

# plt.subplot(1,2,1)
# plt.axis('equal')
# plt.axis('off')
# plt.axis([-1,1,-1,1])
# p.plot('r')
# plot_cmplx(z, 'b')
# plot_cmplx(zi, 'b')
# plot_cmplx(p.vertex, '.y')

# plt.subplot(1,2,2)
# plt.axis('equal')
# plt.axis('off')
# plt.axis([-1,1,-1,1])
# plot_cmplx(m.invmap(z), 'b')
# plot_cmplx(m.invmap(zi.T).T, 'b')
# plot_cmplx(np.exp(theta * 1j), 'r')
# plot_cmplx(m.prevertex, '.y')

# plt.tight_layout()
# plt.savefig('fig1.eps')
# plt.show()



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

# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.imshow(raw_img, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.plot(full_bound_data_xi_w[conf_map_points], full_bound_data_yi_w[conf_map_points], ':o')

# # ax2.imshow(final, cmap=plt.cm.gray)
# # ax2.axis('off')
# ax2.set(xlim=(0, len(orig_img)), ylim=(0, len(orig_img)))
# ax2.invert_yaxis()
# ax2.plot(sorted_points[0], sorted_points[1], '-', label='sorted points')
# ax2.plot(full_bound_data_xi_w[conf_map_points], full_bound_data_yi_w[conf_map_points], ':o')
# ax2.plot(sorted_points[0][top_2_curve[0]], sorted_points[1][top_2_curve[0]], 'o')
# ax2.plot(sorted_points[0][top_2_curve[1]], sorted_points[1][top_2_curve[1]], 'o')
# ax2.legend()
# plt.show()

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