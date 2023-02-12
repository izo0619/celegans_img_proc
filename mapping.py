import matplotlib.pyplot as plt
from skimage import segmentation
from skimage.io import imread
from skimage.morphology import (
    opening, remove_small_objects, area_closing, closing)
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


def do_kdtree(cur_point, original_points):
    """Helper function for sorting a set of points by distance from a specific point

    Args:
        cur_point (list of NDarray): reference point to sort from
        original_points (list of NDarray): points to sort

    Returns:
        sorted_points: array of sorted x, y, and distances
    """
    combined_x_y_arrays = np.dstack(
        [original_points[0].ravel(), original_points[1].ravel()])[0]
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(cur_point)
    return [original_points[0][indexes[0]], original_points[1][indexes[0]], indexes[0], dist]


def plot_cmplx(z, *a, **k):
    """Plots complex array

    Args:
        z (complex array): array to plot
    """
    plt.plot(np.real(z), np.imag(z), *a, **k)


def morphology(orig_img, worm_idx):
    """Perform a set of morphological operations to result in a binary image with 1 worm

    Args:
        orig_img (image): segmented image (ilastik output)
        worm_idx (int): number to specify which worm to use when there are multiple in one image

    Returns:
        final_img : binary segmented image with one worm
    """
    # close any areas smaller than 90 px
    area_closed = area_closing(orig_img, 90)
    opened = opening(area_closed, disk(5))  # opening operation

    # combine layers 2 and 3 and layers 1 and 4
    opened[opened == 1] = 1
    opened[opened == 2] = 0
    opened[opened == 3] = 0
    opened[opened == 4] = 1

    # perform opening again to get rid of any salt in the bg
    fill_worm = label(opened)
    fill_worm = remove_small_objects(fill_worm, 1000)
    fill_worm = closing(fill_worm, disk(15))
    # fill any holes in worm
    fill_worm = ndimage.binary_fill_holes(fill_worm)
    # label the objects in the photo
    fill_worm = label(fill_worm)
    # remove any object that is less than 20,000 px in area
    final = remove_small_objects(fill_worm, 20000)

    # only take the last worm
    unique_worms = np.unique(final)
    for value in unique_worms:
        if value != 0 and value != unique_worms[worm_idx]:
            final[final == value] = 0
    return final


def get_worm_boundary(image):
    """Gets the worm boundary

    Args:
        image (image): binary segmented image

    Returns:
        full_bound_data_xi_w: weighted x boundary
        full_bound_data_yi_w: weighted y boundary
        full_bound_data_xi_w_int: weighted x boundary cast as integer
        full_bound_data_yi_w_int: weighted y boundary cast as integer
        top_2_curve: indices of two points with greatest curvature (essentially endpoints)
    """
    worm_boundary = segmentation.find_boundaries(image, mode='inner')

    pixel_list_row = np.nonzero(worm_boundary)[0]
    pixel_list_col = np.nonzero(worm_boundary)[1]

    original_points = [pixel_list_col, pixel_list_row]
    cur_point = [[original_points[0][0], original_points[1][0]]]
    cur_idx = 0
    sorted_points = [[original_points[0][0]], [original_points[1][0]]]
    tot_dist = 0
    for i in range(len(pixel_list_col)-1):
        original_points[0] = np.delete(original_points[0], cur_idx)
        original_points[1] = np.delete(original_points[1], cur_idx)
        result = do_kdtree(cur_point, original_points)
        tot_dist += result[3]
        if result[3] > ((tot_dist/(i+1))*100):
            break
        sorted_points[0].append(result[0])
        sorted_points[1].append(result[1])
        cur_point = [result[:2]]
        cur_idx = result[2]
        full_bound_data = csaps(range(len(sorted_points[0])), [
                                sorted_points[0], sorted_points[1]], range(len(sorted_points[0])), smooth=1)
        full_bound_data_xi = full_bound_data[0]
        full_bound_data_yi = full_bound_data[1]
    localization = 10
    # x' and y'
    x_t = np.gradient(sorted_points[0][::localization])
    y_t = np.gradient(sorted_points[1][::localization])
    # velocity [x', y']
    vel = np.array([[x_t[i], y_t[i]] for i in range(x_t.size)])
    # speed = sqrt(x'^2 + y'^2) = ds/dt
    speed = np.sqrt(x_t * x_t + y_t * y_t)
    # tangent vector = v/(ds/dt)
    tangent = np.array([1/speed] * 2).transpose() * vel

    # calculate second deriv
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    # calculate curvature
    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / \
        (x_t * x_t + y_t * y_t)**1.5
    full_curvature = []
    j = 0
    for i in range(0, len(full_bound_data_xi), localization):
        if (i + localization > len(full_bound_data_xi)):
            full_curvature.extend(
                [curvature_val[j]] * (len(full_bound_data_xi) - i))
        else:
            full_curvature.extend([curvature_val[j]] * localization)
        j += 1

    # calculate cumulative curvature and find points of local maximum curvature
    full_curvature_cumul = []
    full_curvature_cumul_val = 0
    for i in range(len(curvature_val)):
        full_curvature_cumul_val += curvature_val[i]
        full_curvature_cumul.append(full_curvature_cumul_val)
    fcc_x = range(0, len(full_bound_data_xi), localization)
    fcc_s = CubicSmoothingSpline(
        fcc_x, full_curvature_cumul, smooth=0.00001).spline
    fcc_ds1 = fcc_s.derivative(nu=1)
    fcc_ds2 = fcc_s.derivative(nu=2)
    top_2_curve = (fcc_ds2.roots()[
        (fcc_ds1(fcc_ds2.roots())).argsort()])[-2:].astype(int)
    # sort curvature and split weights
    curve_idx = range(len(full_curvature))
    zipped_cur = zip(full_curvature, curve_idx)
    zipped_cur = sorted(zipped_cur, reverse=False)
    tuples = zip(*zipped_cur)
    full_curve_sorted, curve_idx = [list(tuple) for tuple in tuples]
    portion = round(len(full_curvature)/10)
    weighted_indexes = [curve_idx[x:x+portion]
                        for x in range(0, len(curve_idx), portion)]

    weights = np.ones_like(sorted_points[0]) * 0.5
    w = [0.00000001, 0.000001, 0.000001, 0.000001, 0.000001,
         0.000001, 0.000001, 0.000001, 0.000001, 0.1]
    for i in range(10):
        weights[weighted_indexes[i]] = w[i]

    full_bound_data_w = csaps(range(len(full_bound_data_xi)), [sorted_points[0], sorted_points[1]], range(
        len(full_bound_data_xi)), weights=weights, smooth=0.85)
    full_bound_data_s = CubicSmoothingSpline(range(len(full_bound_data_xi)), [
        sorted_points[0], sorted_points[1]], weights=weights, smooth=0.85).spline
    full_bound_data_xi_w = full_bound_data_w[0]
    full_bound_data_yi_w = full_bound_data_w[1]
    full_bound_data_xi_w_int = full_bound_data_xi_w.astype(int)
    full_bound_data_yi_w_int = full_bound_data_yi_w.astype(int)
    return full_bound_data_xi_w, full_bound_data_yi_w, full_bound_data_xi_w_int, full_bound_data_yi_w_int, top_2_curve


def get_mapping_points(endpoints, full_boundary_length):
    """Gets 48 points around the worm boundary for mapping

    Args:
        endpoints (array): indices of endpoints
        full_boundary_length (int): number of points in full boundary

    Returns:
        conf_map_points: returns 48 indices representing the subset of points to use for mapping
    """
    # find 36 points around worm
    top_2_curve = np.sort(endpoints)
    conf_map_points = [top_2_curve[0]]
    spacing_1 = round((top_2_curve[1]-top_2_curve[0])/36)
    spacing_2 = round(
        (top_2_curve[0] + (full_boundary_length - top_2_curve[1]))/36)
    spacing_level = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16,
                     18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 35.5]
    for i in range(len(spacing_level)):
        conf_map_points.append(
            round(top_2_curve[0] + (spacing_1 * spacing_level[i])))
    conf_map_points.append(top_2_curve[1])
    for i in range(len(spacing_level)):
        conf_map_points.append(
            round(top_2_curve[1] + (spacing_2 * spacing_level[i])) % full_boundary_length)
    return conf_map_points


def get_interior_points(raw_img, boundary_x, boundary_y, mapping_idx):
    """Get all interior points in worm

    Args:
        raw_img (image): raw image
        boundary_x (array): x values of boundary
        boundary_y (array): y values of boundary
        mapping_idx (array): indices of the conformal mapping boundary points

    Returns:
        interior_x: interior x values
        interior_y: interior y values
        colors: color for each point
    """
    # create find interior points
    x, y = np.meshgrid(np.arange(len(raw_img[0])), np.arange(
        len(raw_img)))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    # tup_bounds = tuple(zip(full_bound_data_xi_w_int, full_bound_data_yi_w_int))
    tup_bounds = tuple(
        zip(boundary_x[mapping_idx], boundary_y[mapping_idx]))
    p = Path(tup_bounds)  # make a polygon
    grid = p.contains_points(points)
    # mask with points inside a polygon
    mask = grid.reshape(len(raw_img[0]), len(raw_img))
    mask_points = np.argwhere(mask)
    interior_x = mask_points[:, 1]
    interior_y = mask_points[:, 0]
    interior_points_idx = np.argwhere(grid)
    colors = raw_img.flatten()[interior_points_idx].flatten()
    plt.scatter(interior_x, interior_y)
    plt.show()
    return interior_x, interior_y, colors


def check_direction(switch, boundary_x, boundary_y, mapping_idx, endpoints):
    """Check and change order of the points along with endpoints

    Args:
        switch (bool): reverse or not reverse
        boundary_x (_type_): x values of boundary
        boundary_y (_type_): y values of boundary
        mapping_idx (_type_): indices of conformal mapping boundary
        endpoints (_type_): endpoint indices

    Returns:
        full_worm_ccw_ext: full boundary points in counter clockwise order
    """
    # ** MUST BE COUNTERCLOCKWISE DIRECTION **
    # reverse pts before feed
    if switch:
        full_worm_ccw_ext = np.array(
            [boundary_x[mapping_idx][::-1], boundary_y[mapping_idx][::-1]])
    else:
        full_worm_ccw_ext = np.array(
            [boundary_x[mapping_idx], boundary_y[mapping_idx]])

    plt.scatter(full_worm_ccw_ext[0], full_worm_ccw_ext[1], c=range(
        len(full_worm_ccw_ext[0])), cmap="cool")
    plt.scatter(full_worm_ccw_ext[0], full_worm_ccw_ext[1], c=range(
        len(full_worm_ccw_ext[0])), cmap="cool")
    plt.scatter(full_worm_ccw_ext[0][endpoints[0]],
                full_worm_ccw_ext[1][endpoints[0]], color='red')
    plt.scatter(full_worm_ccw_ext[0][endpoints[1]],
                full_worm_ccw_ext[1][endpoints[1]], color='red')
    plt.show()
    return full_worm_ccw_ext


def perform_mapping(interior_x, interior_y, mapping_points, endpoints, resolution):
    """Perform mapping using stripmap

    Args:
        interior_x (array): x values of interior
        interior_y (array): y values of interior
        mapping_points (array of array):  mapping boundary points
        endpoints (array): indices of endpoints
        resolution (int): resolution of interior mapping

    Returns:
        interior_mapped: location of each interior point post mapping
    """
    p = Polygon(mapping_points[0].tolist(), mapping_points[1].tolist())
    stripmap = Stripmap(p, list(endpoints))
    print("map created")
    x_interior = interior_x.tolist()[::resolution]
    y_interior = interior_y.tolist()[::resolution]
    interior_mapped = stripmap.evalinv(x_interior, y_interior)
    return interior_mapped


def plot_interior_mapped(mapping_points, interior_mapped, colors, resolution):
    """Plot original boundary and mapped interior with corresponding color

    Args:
        mapping_points (array of array): mapping boundary points 
        interior_mapped (array of array): location of each interior point post mapping
        colors (array): color for each point
        resolution (int): resolution of interior mapping
    """
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(mapping_points[0].tolist(), mapping_points[1].tolist())
    axs[0].set_title("original")
    axs[1].scatter(interior_mapped[0], interior_mapped[1],
                   c=colors[::resolution], cmap='gray')
    axs[1].set_title("inverse mapped")
    fig.tight_layout()
    plt.show()


def interpolate_interior(interior_mapped, colors, resolution):
    """Interpolates the colors between known points to generate a final image

    Args:
        interior_mapped (array of array): location of each interior point post mapping
        colors (array): color for each point
        resolution (int): resolution of interior mapping
    """
    nan_idxs = np.argwhere(np.isnan(interior_mapped[0]))
    interior_x = np.delete(interior_mapped[0], nan_idxs)
    interior_y = np.delete(interior_mapped[1], nan_idxs)
    colors = np.delete(colors[::resolution], nan_idxs)
    # set up output to become list of lists
    result = [list(i) for i in zip(interior_x, interior_y)]
    # generate grid data using mgrid
    grid_x, grid_y = np.mgrid[0:25:25000j, 0:1:1000j]
    # grid_a = griddata(result, colors, (grid_x, grid_y), method='cubic')
    grid_b = griddata(result, colors,
                      (grid_x, grid_y), method='linear')
    grid_c = griddata(result, colors,
                      (grid_x, grid_y), method='nearest')

    fig, axs = plt.subplots(2, 1)
    axs[1].imshow(grid_b.T, cmap='gray')
    axs[1].set_title("linear")
    axs[0].imshow(grid_c.T, cmap='gray')
    axs[0].set_title("nearest")
    fig.tight_layout()
    plt.show()
    print("complete")
