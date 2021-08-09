import sys; sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np

from conformalmapping import *
G = Splinep.from_complex_list([ 
    0.2398 + 0.6023j, 0.3567 + 1.0819j, 0.2632 + 1.5965j,
    -0.5205 + 1.7485j, -1.0585 + 1.1170j, -1.0702 + 0.5088j,
    -0.5906 + 0.0994j, -0.7778 - 0.4269j, -1.2924 - 0.6140j,
    -1.4561 - 1.2456j, -0.5439 - 1.3509j, 0.2515 - 1.0702j,
    0.3099 - 0.6023j, 0.7427 - 0.5906j, 1.1053 - 0.1813j,
    1.2807 + 0.3567j 
])
sm = SzMap(G, 0)
sm.plot()
S = Szego(G, 0)
t = np.arange(20)/20.
cl = np.array([ 
    0.2398 + 0.6023j, 0.3567 + 1.0819j, 0.2632 + 1.5965j,
    -0.5205 + 1.7485j, -1.0585 + 1.1170j, -1.0702 + 0.5088j,
    -0.5906 + 0.0994j, -0.7778 - 0.4269j, -1.2924 - 0.6140j,
    -1.4561 - 1.2456j, -0.5439 - 1.3509j, 0.2515 - 1.0702j,
    0.3099 - 0.6023j, 0.7427 - 0.5906j, 1.1053 - 0.1813j,
    1.2807 + 0.3567j 
])

plt.subplot(1,2,1)
# G.plot()
plt.plot(cl.real, cl.imag, ':o')
# zs = G(t)
# plt.plot(zs.real, zs.imag, 'ro')
# plt.plot(zs.real[0], zs.imag[0], 'bo')
# plt.plot(zs.real[1], zs.imag[1], 'bo')
# plt.gca().set_aspect('equal')
# plt.gca().axis(G.plotbox())

plt.subplot(1,2,2)
c = Circle(0, 1)
c.plot()
zs = np.exp(1.0j * S.theta(t))
plt.plot(zs.real, zs.imag, 'ro')
plt.plot(zs.real[0], zs.imag[0], 'bo')
plt.plot(zs.real[1], zs.imag[1], 'bo')
plt.gca().set_aspect('equal')
plt.gca().axis(c.plotbox())
plt.show()