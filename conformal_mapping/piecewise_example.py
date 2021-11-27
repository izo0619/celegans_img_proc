import sys; sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from aaapy.aaa import aaa

from conformalmapping import *

## 1. create 20 individual squares
## 2. use szmap to boundary map
## 3. try aaa (1000 interior points)

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

length = 10
num_pts = 10
all_squares = np.array([])
for i in range(20):
    line_a = length*i + np.linspace(length/2, -length/2,num_pts, endpoint=False) * 1j
    line_b = np.linspace(length*i,length*(i+1),num_pts, endpoint=False) + (-length/2)*1j
    line_c = length*(i+1) + np.linspace(-length/2, length/2, num_pts, endpoint=False) * 1j
    line_d = np.linspace(length*(i+1), length*i,num_pts, endpoint=True) + (length/2)*1j
    L_pts = ((np.concatenate((line_a, line_b, line_c, line_d))))
    L_spline = Splinep.from_complex_list(L_pts)

    glines = []
    gline_a = length*(i+0.5) + np.linspace(length/2, -length/2,num_pts) * 1j
    glines.append(gline_a)
    gline_b = np.linspace(length*i,length*(i+1),num_pts) + 0j
    glines.append(gline_b)
    gline_c = np.linspace(length*i,length*(i+1), num_pts) \
            + np.linspace(length/2, -length/2, num_pts) * 1j
    glines.append(gline_c)
    gline_d = np.linspace(length*i,length*(i+1), num_pts) \
            + np.linspace(-length/2, length/2, num_pts) * 1j
    glines.append(gline_d)

    conformalCenter = length*(i+0.5) + 0j
    plt.plot(conformalCenter.real, conformalCenter.imag, marker='x', color='blue')
    sm = SzMap(L_spline, conformalCenter)
    S = Szego(L_spline, conformalCenter)
    t = np.arange(1000)/1000.
    L_zs_orig = L_spline(t)
    L_zs_circle = np.exp(1.0j * S.theta(t))
    # reverse map: circle -> shape 
    s = aaa(L_zs_orig, L_zs_circle)
    # forward map: shape -> circle 
    r = aaa(L_zs_circle, L_zs_orig, tol=1e-7)

    rev_lines = np.array([])
    circ_lines = np.array([])
    # turn square gridlines into circles
    g_lines_all = np.array([])
    for line in glines:
        circ_line = r(line)
        rev_line = s(circ_line)
        circ_lines = np.concatenate((circ_lines, circ_line))
        rev_lines = np.concatenate((rev_lines, rev_line))
        g_lines_all = np.concatenate((g_lines_all, line))
    
    # plt.subplot(2, 1, 1)
    # plt.scatter(circ_lines.real, circ_lines.imag)
    # plt.subplot(2, 1, 2)
    plt.scatter(g_lines_all.real, g_lines_all.imag, facecolors='none', edgecolors='r')
    plt.scatter(rev_lines.real, rev_lines.imag, marker="x")

    all_squares = np.concatenate((all_squares, L_pts))

# plt.subplot(2, 1, 2) # plot 2: squares 
plt.plot(all_squares.real, all_squares.imag)
# plt.plot(all_squares.real, all_squares.imag)
plt.show()
# L_spline = Splinep.from_complex_list(all_squares)
# # conformal mapping
# conformalCenter = 0.7 + 0.7j
# sm = SzMap(L_spline, conformalCenter)
# S = Szego(L_spline, conformalCenter)
# # lay down 3000 evenly spaced points along the spline
# t = np.arange(3000)/3000.
# L_zs_orig = L_spline(t)
# # find corresponding circle points
# L_zs_circle = np.exp(1.0j * S.theta(t))

# # reverse map: circle -> shape 
# s = aaa(L_zs_orig, L_zs_circle)
# # forward map: shape -> circle 
# r = aaa(L_zs_circle, L_zs_orig, tol=1e-7)
# gd = unitdisk().grid()
# lst = []
# for curve in gd.curves:
#     newcurve = s(curve)
#     # newcurve = sm.applyMap(curve)
#     lst.append(newcurve)
# gc_shape = GridCurves(lst)
# gc_circle = GridCurves(gd.curves)

# # shape plot
# plt.subplot(1,2,1)
# plt.scatter(L_zs_orig.real, L_zs_orig.imag, c=t,cmap="cool", s=30)
# gc_shape.plot()

# # circle plot
# plt.subplot(1,2,2)
# c = Circle(0, 1)
# c.plot()
# gc_circle.plot()
# plt.scatter(L_zs_circle.real, L_zs_circle.imag, c=t,cmap="cool", s=30)

# # extra plotting features
# plt.gca().set_aspect('equal')
# plt.show()