import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import griddata


#define a function
def func(x,y):
    return (x**2+y**2+(x*y)**2)**2

#generate grid data using mgrid
grid_x,grid_y = np.mgrid[0:1:1000j, 0:1:2000j]

#generate random points
rng = np.random.default_rng()
points = rng.random((1000, 2))


#generate values from the points generated above
values = func(points[:,0], points[:,1])

#generate grid data using the points and values above
grid_a = griddata(points, values, (grid_x, grid_y), method='cubic')

grid_b = griddata(points, values, (grid_x, grid_y), method='linear')

grid_c = griddata(points, values, (grid_x, grid_y), method='nearest')


#visualizations
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(func(grid_x,grid_y))
axs[0, 0].set_title("main")
axs[1, 0].plot(grid_a)
axs[1, 0].set_title("cubic")
axs[0, 1].plot(grid_b)
axs[0, 1].set_title("linear")
axs[1, 1].plot(grid_c)
axs[1, 1].set_title("nearest")
fig.tight_layout()
plt.show()