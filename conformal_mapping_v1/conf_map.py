import matlab.engine
import matlab
import matplotlib.pyplot as plt
import numpy as np

# clockwise
full_worm_cw = np.load("wormfullbound.npy")
full_worm_ends_cw = [83, 1413]
full_worm_cw = full_worm_cw[0] + 1j*full_worm_cw[1]
# ccw
full_worm_ccw = np.load("wormfullboundccw.npy")
full_worm_ends_ccw = [1, 1321]
# plt.scatter(full_worm_ccw[0], full_worm_ccw[1], c=range(len(full_worm_ccw[0])), cmap="cool")
# plt.scatter(full_worm_ccw[0][full_worm_ends_ccw[0]], full_worm_ccw[1][full_worm_ends_ccw[0]])
# plt.scatter(full_worm_ccw[0][full_worm_ends_ccw[1]], full_worm_ccw[1][full_worm_ends_ccw[1]])
# plt.show()
full_worm_ccw = full_worm_ccw[0] + 1j*full_worm_ccw[1]

eng = matlab.engine.start_matlab()
input_arr = matlab.double(eng.cell2mat(full_worm_ccw.tolist()), is_complex=True)
input_bounds = matlab.double(eng.cell2mat(full_worm_ends_ccw), is_complex=True)
result = eng.sc_strip_map(input_arr, input_bounds)
print("complete")

# ret = eng.triarea(1.0,5.0)
# print(ret)

# ret = eng.linspace(1.0,5.0)
# print(ret)