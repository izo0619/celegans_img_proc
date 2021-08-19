import sys; sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np

from conformalmapping import *
# G = Splinep.from_complex_list([ 
#     0.2398 + 0.6023j, 0.3567 + 1.0819j, 0.2632 + 1.5965j,
#     -0.5205 + 1.7485j, -1.0585 + 1.1170j, -1.0702 + 0.5088j,
#     -0.5906 + 0.0994j, -0.7778 - 0.4269j, -1.2924 - 0.6140j,
#     -1.4561 - 1.2456j, -0.5439 - 1.3509j, 0.2515 - 1.0702j,
#     0.3099 - 0.6023j, 0.7427 - 0.5906j, 1.1053 - 0.1813j,
#     1.2807 + 0j 
# ])
G = Splinep.from_complex_list([ 
    0.2398 + 0.6023j, 0.3567 + 1.0819j, 0.2632 + 1.5965j,
    -0.5205 + 1.7485j, -1.0585 + 1.1170j, -1.0702 + 0.5088j,
    -0.5906 + 0.0994j, -0.7778 - 0.4269j, -1.2924 - 0.6140j,
    -1.4561 - 1.2456j, -0.5439 - 1.3509j, 0.2515 - 1.0702j,
    0.3099 - 0.6023j, 0.7427 - 0.5906j, 1.1053 - 0.1813j,
    1.2807 + 0.3567j 
])
# G = Splinep.from_complex_list([ 
#     # -1 + 1j, 0+ 1.5j, 1 + 1j, 0.5 + 0j, 1 - 1j, 0 - 1.5j, -1 - 1j, -0.5 + 0j
#     -1 + 1j, -0.5 + 0j, -1 - 1j, 0 - 1.5j, 1 - 1j, 0.5 + 0j, 1 + 1j, 0 + 1.5j
# ])
# G = Splinep.from_complex_list([ 
#     -1 + 1j, 0+ 1.5j, 1 + 1j, 0.5 + 0j, 1 - 1j, 0 - 1.5j, -1 - 1j, -0.5 + 0j
# ])
# G = Splinep.from_complex_list([ 
#     -1 + 1j, 1 + 1j, 1 - 1j, -1 - 1j
# ])
# G = Splinep.from_complex_list([ 
#     -1 + 1j, -1 - 1j, 1 - 1j, 1 + 1j
# ])
# G = Splinep.from_complex_list([ 
#     -1 + 1j, -1 + 0j, -1 - 1j, 0 - 1j, 1 - 1j, 1 + 0j, 1 + 1j, 0 + 1j
# ])
# G = Splinep.from_complex_list([ 
#     -1 + 1j, 0 + 1j, 1 + 1j, 1 + 0j, 1 - 1j, 0 - 1j, -1 - 1j, -1 + 0j
# ])
sm = SzMap(G, 0)
sm.plot()
S = Szego(G, 0)
t = np.arange(30)/30.
cl = np.array([ 
    0.2398 + 0.6023j, 0.3567 + 1.0819j, 0.2632 + 1.5965j,
    -0.5205 + 1.7485j, -1.0585 + 1.1170j, -1.0702 + 0.5088j,
    -0.5906 + 0.0994j, -0.7778 - 0.4269j, -1.2924 - 0.6140j,
    -1.4561 - 1.2456j, -0.5439 - 1.3509j, 0.2515 - 1.0702j,
    0.3099 - 0.6023j, 0.7427 - 0.5906j, 1.1053 - 0.1813j,
    1.2807 + 0.3567j 
])

plt.subplot(1,2,1)
G.plot()
# plt.plot(cl.real, cl.imag, ':o')

zs = G(t)
np.set_printoptions(precision=4, suppress=True, linewidth=15)
N = 512
th= 2*np.pi*np.arange(N)/float(N)
t_2 = S.invtheta(th)
w = G(t_2)
c = np.fft.fft(w)/float(N)
f = lambda z : np.polyval(helpers.flipud(c),z)
gd = unitdisk().grid()
lst = []
for curve in gd.curves:
    newcurve = f(curve)
    # newcurve = sm.applyMap(curve)
    lst.append(newcurve)
gc = GridCurves(lst)
gc_orig = GridCurves(gd.curves)
# gc_orig.plot()
# gc.plot()
# G.plot()
# plt.gca().set_aspect('equal')
# plt.gca().axis(G.plotbox())
# ax = plt.gca()
# ax.set_xticks([]) 
# ax.set_yticks([]) 
# plt.show()


# plt.plot(zs.real, zs.imag, 'ro')
# plt.scatter(zs.real, zs.imag, c=t, cmap="cool", s=30)
plt.plot()
plt.plot(sm.applyMap([0.1 + 0.5j]).real, sm.applyMap([0.1 + 0.5j]).imag, 'bo')
plt.plot(sm.applyMap([0.3+0.2j]).real, sm.applyMap([0.3+0.2j]).imag, 'mo')
plt.plot(sm.applyMap([0 + 1j]).real, sm.applyMap([0 + 1j]).imag, 'go')
plt.plot(sm.applyMap([1+0j]).real, sm.applyMap([1+0j]).imag, 'ro')
plt.plot(sm.applyMap(zs[0]).real, sm.applyMap(zs[0]).imag, 'yo')
plt.plot(sm.applyMap([2+0j]).real, sm.applyMap([2+0j]).imag, 'co')
gc.plot()
# plt.plot(zs.real[0], zs.imag[0], 'bo', fillstyle='none')
# plt.plot(zs.real[1], zs.imag[1], 'bo', fillstyle='none')
plt.gca().set_aspect('equal')
plt.gca().axis(G.plotbox())
print(sm.applyMap([2+0j]))
plt.subplot(1,2,2)
c = Circle(0, 1)
c.plot()
gc_orig.plot()

zs_2 = np.exp(1.0j * S.theta(t))
plt.plot(0.1, 0.5, 'bo')
plt.plot(0.3, 0.2, 'mo')
plt.plot(0, 1, 'go')
plt.plot(1, 0, 'ro')
plt.plot(zs.real[0], zs.imag[0], 'yo')
plt.plot(2, 0, 'co')
# plt.plot(zs.real, zs.imag, 'ro')
# plt.scatter(zs_2.real, zs_2.imag, c=t, cmap="cool", s=30)
# plt.plot(zs.real[0], zs.imag[0], 'bo', fillstyle='none')
# plt.plot(zs.real[1], zs.imag[1], 'bo', fillstyle='none')
plt.gca().set_aspect('equal')
plt.gca().axis(c.plotbox())
plt.show()