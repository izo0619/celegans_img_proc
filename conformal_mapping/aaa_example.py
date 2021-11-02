import sys; sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from aaapy.aaa import aaa

from conformalmapping import *

## 1. place 3000 points along L shape
## 2. use szmap to boundary map
## 3. try aaa (1000 interior points)

### does boundary map have origin? how to adjust origin?

# create L shape
num_pts = 100
line_a = 0 + np.linspace(2,0,num_pts, endpoint=False) * 1j
line_b = np.linspace(0,2,num_pts, endpoint=False) + 0j
line_c = 2 + np.linspace(0,1,num_pts, endpoint=False) * 1j
line_d = np.linspace(2,1,num_pts, endpoint=False) + 1j
line_e = 1 + np.linspace(1,2,num_pts, endpoint=False) * 1j
line_f = np.linspace(1,0,num_pts) + 2j

# concatenate all edges of the L
L_pts = ((np.concatenate((line_a, line_b, line_c, line_d, line_e, line_f))))
# create spline 
L_spline = Splinep.from_complex_list(L_pts)

# conformal mapping
conformalCenter = 0.7 + 0.7j
sm = SzMap(L_spline, conformalCenter)
S = Szego(L_spline, conformalCenter)
# lay down 3000 evenly spaced points along the spline
t = np.arange(3000)/3000.
L_zs_orig = L_spline(t)
# find corresponding circle points
L_zs_circle = np.exp(1.0j * S.theta(t))

# reverse map: circle -> shape 
s = aaa(L_zs_orig, L_zs_circle)
# forward map: shape -> circle 
r = aaa(L_zs_circle, L_zs_orig, tol=1e-7)
gd = unitdisk().grid()
lst = []
for curve in gd.curves:
    newcurve = s(curve)
    # newcurve = sm.applyMap(curve)
    lst.append(newcurve)
gc_shape = GridCurves(lst)
gc_circle = GridCurves(gd.curves)

# shape plot
plt.subplot(1,2,1)
plt.scatter(L_zs_orig.real, L_zs_orig.imag, c=t,cmap="cool", s=30)
gc_shape.plot()

# circle plot
plt.subplot(1,2,2)
c = Circle(0, 1)
c.plot()
gc_circle.plot()
plt.scatter(L_zs_circle.real, L_zs_circle.imag, c=t,cmap="cool", s=30)

# extra plotting features
plt.gca().set_aspect('equal')
plt.show()