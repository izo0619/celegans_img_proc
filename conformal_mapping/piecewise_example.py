import sys; sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from aaapy.aaa import aaa

from conformalmapping import *

## 1. create 20 individual squares
## 2. use szmap to boundary map
## 3. try aaa (1000 interior points)

def complex_dist(pt1, pt2):
    return np.sqrt((np.square(pt1.real - pt2.real)) + (np.square(pt1.imag - pt2.imag)))
### 

# # create square shape
# num_pts = 100
# length = 64
# line_a = -length + np.linspace(1,-1,num_pts, endpoint=False) * 1j
# line_b = np.linspace(-length,length,num_pts, endpoint=False) + -1j
# line_c = length + np.linspace(-1,1,num_pts, endpoint=False) * 1j
# line_d = np.linspace(length,-length,num_pts, endpoint=False) + 1j

# # concatenate all edges of the L
# L_pts = ((np.concatenate((line_a, line_b, line_c, line_d))))
# # create spline 
# L_spline = Splinep.from_complex_list(L_pts)


# #### generate square by square
# length = 10
# num_pts = 10
# all_squares = np.array([])
# for i in range(20):
#     line_a = length*i + np.linspace(length/2, -length/2,num_pts, endpoint=False) * 1j
#     line_b = np.linspace(length*i,length*(i+1),num_pts, endpoint=False) + (-length/2)*1j
#     line_c = length*(i+1) + np.linspace(-length/2, length/2, num_pts, endpoint=False) * 1j
#     line_d = np.linspace(length*(i+1), length*i,num_pts, endpoint=True) + (length/2)*1j
#     L_pts = ((np.concatenate((line_a, line_b, line_c, line_d))))
#     L_spline = Splinep.from_complex_list(L_pts)

#     glines = []
#     gline_a = length*(i+0.5) + np.linspace(length/2, -length/2,num_pts) * 1j
#     glines.append(gline_a)
#     gline_b = np.linspace(length*i,length*(i+1),num_pts) + 0j
#     glines.append(gline_b)
#     gline_c = np.linspace(length*i,length*(i+1), num_pts) \
#             + np.linspace(length/2, -length/2, num_pts) * 1j
#     glines.append(gline_c)
#     gline_d = np.linspace(length*i,length*(i+1), num_pts) \
#             + np.linspace(-length/2, length/2, num_pts) * 1j
#     glines.append(gline_d)

#     conformalCenter = length*(i+0.5) + 0j
#     plt.plot(conformalCenter.real, conformalCenter.imag, marker='x', color='blue')
#     sm = SzMap(L_spline, conformalCenter)
#     S = Szego(L_spline, conformalCenter)
#     t = np.arange(1000)/1000.
#     L_zs_orig = L_spline(t)
#     L_zs_circle = np.exp(1.0j * S.theta(t))
#     # reverse map: circle -> shape 
#     s = aaa(L_zs_orig, L_zs_circle)
#     # forward map: shape -> circle 
#     r = aaa(L_zs_circle, L_zs_orig, tol=1e-7)

#     rev_lines = np.array([])
#     circ_lines = np.array([])
#     # turn square gridlines into circles
#     g_lines_all = np.array([])
#     for line in glines:
#         circ_line = r(line)
#         rev_line = s(circ_line)
#         circ_lines = np.concatenate((circ_lines, circ_line))
#         rev_lines = np.concatenate((rev_lines, rev_line))
#         g_lines_all = np.concatenate((g_lines_all, line))
    
#     plt.scatter(g_lines_all.real, g_lines_all.imag, facecolors='none', edgecolors='r')
#     plt.scatter(rev_lines.real, rev_lines.imag, marker="x")

#     all_squares = np.concatenate((all_squares, L_pts))

# # plt.subplot(2, 1, 2) # plot 2: squares 
# plt.plot(all_squares.real, all_squares.imag)
# # plt.plot(all_squares.real, all_squares.imag)
# plt.show()

#### global gridlines
def calculate_global_error(num_boxes, res):
    length = 200 # length of whole object - long way
    width = 10 # width of whole object - short way
    num_pts = 5 # density of border pts
    # num_boxes = 10
    box_length = length/num_boxes
    all_squares = np.array([])
    all_squares2 =[]

    # create general boundary
    line_a = np.linspace(width/2,-width/2,num_pts, endpoint=False) * 1j
    line_b = np.linspace(0,length,num_pts, endpoint=False) + -(width/2)*1j
    line_c = length + np.linspace(-width/2,width/2,num_pts, endpoint=False) * 1j
    line_d = np.linspace(length,0,num_pts, endpoint=False) + (width/2)*1j
    L_pts = ((np.concatenate((line_a, line_b, line_c, line_d))))
    L_spline = Splinep.from_complex_list(L_pts)

    # create gridlines
    glines = []
    for j in np.linspace(-width/2,width/2, int(width/res), endpoint=False):
        for i in np.linspace(0,length, int(length/res), endpoint=False):
            coord = (i+(res/2)) + (j+(res/2))*1j
            glines.append(coord)
            # np.append(glines, coord)
    glines = np.array(glines)

    # generates array of forward and reverse maps
    reverse_maps = []
    forward_maps = []
    for i in range(num_boxes):
        line_a = box_length*i + np.linspace(width/2, -width/2,num_pts, endpoint=False) * 1j
        line_b = np.linspace(box_length*i,box_length*(i+1),num_pts, endpoint=False) + (-width/2)*1j
        line_c = box_length*(i+1) + np.linspace(-width/2, width/2, num_pts, endpoint=False) * 1j
        line_d = np.linspace(box_length*(i+1), box_length*i,num_pts, endpoint=True) + (width/2)*1j
        L_pts = ((np.concatenate((line_a, line_b, line_c, line_d))))
        L_spline = Splinep.from_complex_list(L_pts)

        conformalCenter = box_length*(i+0.5) + 0j
        #plt.plot(conformalCenter.real, conformalCenter.imag, marker='x', color='gray')
        sm = SzMap(L_spline, conformalCenter)
        S = Szego(L_spline, conformalCenter)
        t = np.arange(1000)/1000.
        L_zs_orig = L_spline(t)
        L_zs_circle = np.exp(1.0j * S.theta(t))
        # reverse map: circle -> shape 
        s = aaa(L_zs_orig, L_zs_circle)
        # forward map: shape -> circle 
        r = aaa(L_zs_circle, L_zs_orig, tol=1e-7)
        reverse_maps.append(s)
        forward_maps.append(r)
        all_squares = np.concatenate((all_squares, L_pts))
        all_squares2.append(L_pts)

    glines_remap = []
    global_error = 0
    error_offset = 0
    all_errors = []
    test_x=[]
    test_y=[]
    for pt in glines:
        box_num = int(pt.real // box_length)
        # last edge scenario
        if box_num == num_boxes: box_num -= 1
        # double map scenario
        elif pt.real % box_length == 0 and box_num != 0:
            error_offset -= 1
            break
            # new_pt = forward_maps[box_num-1](pt) # forward map
            # new_pt = reverse_maps[box_num-1](new_pt) # reverse map
            # glines_remap.append(new_pt)
            # global_error += complex_dist(pt, new_pt)
        new_pt = forward_maps[box_num](pt) # forward map
        test_x.append(new_pt.real)
        test_y.append(new_pt.imag)
        new_pt = reverse_maps[box_num](new_pt) # reverse map
        glines_remap.append(new_pt)
        all_errors.append(complex_dist(pt, new_pt))
        global_error += complex_dist(pt, new_pt)

    glines_remap = np.array(glines_remap)
    global_error = global_error / (len(glines) + error_offset)

    ratio = length / (width * num_boxes)
    name = "X:Y ratio of " + str(ratio) + " at " + str(res) + " resolution"
    # file.write(str(ratio) + "," + str(global_error))
    # file.write("\n")

    # plt.scatter(glines.real, glines.imag, c='red', marker="D", label="Original gridlines")
    fig, axs = plt.subplots(3, 1, figsize=(16,9))
    axs[0].plot(all_squares.real, all_squares.imag, color="gray", label="Boundary lines")
    # axs[0].scatter(glines.real, glines.imag, c='red', marker="D", label="Original gridlines")
    scatter = axs[0].scatter(glines_remap.real, glines_remap.imag, c=all_errors, cmap="spring", label="Remapped gridlines")
    plt.colorbar(scatter, format='%.2e', ax=axs[0])
    axs[0].legend(loc="upper right")
    axs[0].set_title(name)
    
    axs[1].hist(all_errors)
    axs[1].set_xlabel('error')
    axs[1].set_ylabel('frequency')
    axs[1].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.tight_layout()
    for i in range(len(all_squares2)):
        pain = forward_maps[i](all_squares2[i])
        axs[2].plot(pain.real, pain.imag)
    plt.show()
    # plt.savefig("plots/" + str(ratio) + "_ratio_" + str(res) + "_res" + ".png", dpi=200)

# to_run = [1,2,4,10,20,40,100, 200, 400]
# file = open('global_error_results.txt', 'w')
# for nb in to_run:
#     calculate_global_error(nb, 1)
# file.close()

calculate_global_error(4, 1)