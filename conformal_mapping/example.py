import sys; sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from aaapy.aaa import aaa

from conformalmapping import *
# G = Splinep.from_complex_list([ 
#     0.2398 + 0.6023j, 0.3567 + 1.0819j, 0.2632 + 1.5965j,
#     -0.5205 + 1.7485j, -1.0585 + 1.1170j, -1.0702 + 0.5088j,
#     -0.5906 + 0.0994j, -0.7778 - 0.4269j, -1.2924 - 0.6140j,
#     -1.4561 - 1.2456j, -0.5439 - 1.3509j, 0.2515 - 1.0702j,
#     0.3099 - 0.6023j, 0.7427 - 0.5906j, 1.1053 - 0.1813j,
#     1.2807 + 0j 
# ])
# G = Splinep.from_complex_list([ 
#     0.2398 + 0.6023j, 0.3567 + 1.0819j, 0.2632 + 1.5965j,
#     -0.5205 + 1.7485j, -1.0585 + 1.1170j, -1.0702 + 0.5088j,
#     -0.5906 + 0.0994j, -0.7778 - 0.4269j, -1.2924 - 0.6140j,
#     -1.4561 - 1.2456j, -0.5439 - 1.3509j, 0.2515 - 1.0702j,
#     0.3099 - 0.6023j, 0.7427 - 0.5906j, 1.1053 - 0.1813j,
#     1.2807 + 0.3567j 
# ])
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

# skinny diagonal 
# G = Splinep.from_complex_list([ 
#     -3 + 3j, -.1 - .1j, 3 - 3j, .1 + .1j
# ])
G = Splinep.from_complex_list([ 
    -3 + 3j, .1 + .1j, 3 - 3j, -.1 - .1j
])
# # skinny vertical
# G = Splinep.from_complex_list([ 
#     0 + 0.12j, -1 + 0j,  0 - 0.12j, 1 + 0j,
# ])
# skinny rectangle
# G = Splinep.from_complex_list([ 
#     0 + 0.2j, -1.5 + 0.2j, -2 + 0.2j, -2 - 0.2j, -1.5 - 0.2j, 0 - 0.2j, 1.5 - 0.2j, 2 - 0.2j, 2 + 0.2j, 1.5 + 0.2j
# ])
# u shape??
# G = Splinep.from_complex_list([ 
#     -2 + 4j, 0 + 1j, 2 + 4j, 1 + 3.5j, 0 + 2.5j, -1 + 3.5j
# ])
# G_lst = np.array([ 
#     -2 + 1j, 0 - 2j, 2 + 1j, 1 + 0.5j, 0 - 0.5j, -1 + 0.5j
# ])
# G = Splinep.from_complex_list([ 
#     0 + 2j, 0 + 0j, 2 + 0j, 2.5 + 1j, 1 +1j, 0.7 + 1.7j
# ])
# G = Splinep.from_complex_list([ 
#     2 + 1j, 1 + 1j, 1 + 2j, 0 + 2j, 0 + 0j, 2 + 0j
# ])
# G = Splinep.from_complex_list([ 
#     2 + 1j, 2 + 0j, 0 +0j, 0 + 2j, 1 + 2j, 1 + 1j, 
# ])
# G = Splinep.from_complex_list([ 
#     -1 + 1j, 0 + 1j, 1 + 1j, 1 + 0j, 1 - 1j, 0 - 1j, -1 - 1j, -1 + 0j
# ])
sm = SzMap(G, 0)
sm.plot()
S = Szego(G, 0)
t = np.arange(500)/500.
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
# plt.plot(cl.real, cl.imag, ':o')

# zs = G(t)
# zs_2 = np.exp(1.0j * S.theta(t))

# zs = np.array([ 
#     2 + 1j, 1 + 1j, 1 + 2j, 0 + 2j, 0 +0j, 2 + 0j
# ])
# zs_2 = np.array([0.9908 + 0.1356j,0.2582 + 0.9661j, -0.7910 + 0.6119j, -0.8666 + 0.4990j, -0.2582 - 0.9661j, 1.0000 + 0.0000j])
zs = np.array([1+1j, 0.5 + 1j, 0 + 1j, -0.5 + 1j,
            -1+1j, -1 + 0.5j, -1 + 0j, -1 - 0.5j,
            -1-1j, -0.5 - 1j, 0 - 1j, 0.5 - 1j, 
            1-1j, 1 - 0.5j, 1 + 0j, 1 + 0.5j])
zs_2 = np.array([0.0000 + 1.0000j,  -0.5 + 0.867j, -0.71 + 0.71j, -0.867 + 0.5j,
                -1.0000 + 0.0000j,  -0.867 - 0.5j, -0.71 -0.71j, -0.5 -0.867j, 
                0.0000 - 1.0000j, 0.5 - 0.867j, 0.71 - 0.71j, 0.867 - 0.5j,
                1.0000 + 0.0000j, 0.867 + 0.5j, 0.71 + 0.71j, 0.5 + 0.867j])

# lines inside square
line = np.linspace(-1,1,10) + 0.1j
line2 = np.linspace(-1,1,10) * 1j + 0.1
line6 = np.linspace(1,-1,10) + np.linspace(-1,1,10) * 1j
line7 = np.linspace(-1,1,10) + np.linspace(-1,1,10) * 1j

# lines inside circle
line3 = np.linspace(-1,1,10) *np.exp(1j*np.pi/3)
line4 = np.linspace(-1,1,10) *np.exp(1j*np.pi/6)
line5 = np.linspace(-1, 1, 10) * np.exp(1j*np.pi/2)
# np.set_printoptions(precision=4, suppress=True, linewidth=15)
# N = 512
# th= 2*np.pi*np.arange(N)/float(N)
# t_2 = S.invtheta(th)
# w = G(t_2)
# c = np.fft.fft(w)/float(N)
# f = lambda z : np.polyval(helpers.flipud(c),z)
# s = aaa(zs, zs_2, tol=1e-7, mmax=500)
s = aaa(zs, zs_2)
r = aaa(zs_2, zs, tol=1e-7)
gd = unitdisk().grid()
lst = []
for curve in gd.curves:
    newcurve = r(curve)
    # newcurve = sm.applyMap(curve)
    lst.append(newcurve)
gc = GridCurves(lst)
gc_orig = GridCurves(gd.curves)
# gc_orig.plot()
# gc.plot()
# G.plot()
plt.gca().set_aspect('equal')
plt.gca().axis(G.plotbox())
# ax = plt.gca()
# ax.set_xticks([]) 
# ax.set_yticks([]) 
# plt.show()

fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
# square to circle
axs[0].plot(r(zs).real, r(zs).imag, 'r:')
axs[0].plot(r(line).real, r(line).imag, 'b:')
axs[0].plot(r(line2).real, r(line2).imag, 'c:')
axs[0].plot(r(line6).real, r(line6).imag, 'm:')
axs[0].plot(r(line7).real, r(line7).imag, 'y:')
# plt.plot(line.real, line.imag, 'b:')
# plt.plot(line2.real, line2.imag, 'c:')
# plt.plot(line6.real, line6.imag, 'm:')
# plt.plot(line7.real, line7.imag, 'y:')
# plt.show()

# circle to square
# axs[1].plot(s(zs_2).real, s(zs_2).imag, 'r-')
# axs[1].plot(s(line3).real, s(line3).imag, 'b-')
# axs[1].plot(s(line4).real, s(line4).imag, 'c-')
# axs[1].plot(s(line5).real, s(line5).imag, 'm-')

# input square lines 
axs[1].plot(zs.real, zs.imag, 'r:')
axs[1].plot(line.real, line.imag, 'b:')
axs[1].plot(line2.real, line2.imag, 'c:')
axs[1].plot(line6.real, line6.imag, 'm:')
axs[1].plot(line7.real, line7.imag, 'y:')

plt.show()

# # plt.plot(zs.real, zs.imag, 'ro')
# # plt.scatter(zs.real, zs.imag, c=t, cmap="cool", s=30)
# plt.plot()
# # plt.plot(sm.applyMap([0.1 + 0.5j]).real, sm.applyMap([0.1 + 0.5j]).imag, 'bo')
# # plt.plot(sm.applyMap([0.3+0.2j]).real, sm.applyMap([0.3+0.2j]).imag, 'mo')
# # plt.plot(sm.applyMap([0 + 1j]).real, sm.applyMap([0 + 1j]).imag, 'go')
# # plt.plot(sm.applyMap([1+0j]).real, sm.applyMap([1+0j]).imag, 'ro')
# # plt.plot(sm.applyMap(zs[0]).real, sm.applyMap(zs[0]).imag, 'yo')
# # plt.plot(sm.applyMap([2+0j]).real, sm.applyMap([2+0j]).imag, 'co')
# # plt.scatter(s(zs_2).real, s(zs_2).imag, cmap="cool", s=30)
# plt.plot(r(zs_2).real, r(zs_2).imag, '-')
# print(r(zs))
# # plt.plot(G_lst.real, G_lst.imag, 'co')
# gc.plot()
# # plt.plot(0.7, 0.7, 'co')
# # plt.plot(zs.real[0], zs.imag[0], 'bo', fillstyle='none')
# # plt.plot(zs.real[1], zs.imag[1], 'bo', fillstyle='none')
# plt.gca().set_aspect('equal')
# plt.gca().axis(G.plotbox())
# plt.subplot(1,2,2)
# c = Circle(0, 1)
# c.plot()
# gc_orig.plot()

# # zs_2 = np.exp(1.0j * S.theta(t))
# # plt.plot(0.1, 0.5, 'bo')
# # plt.plot(0.3, 0.2, 'mo')
# # plt.plot(0, 1, 'go')
# # plt.plot(1, 0, 'ro')
# # plt.plot(zs.real[0], zs.imag[0], 'yo')
# # plt.plot(2, 0, 'co')
# # plt.scatter(zs_2.real, zs_2.imag, cmap="cool", s=30)
# plt.plot(zs_2.real, zs_2.imag, '-')
# # plt.plot(zs.real, zs.imag, 'ro')
# # plt.scatter(zs_2.real, zs_2.imag, c=t, cmap="cool", s=30)
# # plt.plot(zs.real[0], zs.imag[0], 'bo', fillstyle='none')
# # plt.plot(zs.real[1], zs.imag[1], 'bo', fillstyle='none')
# plt.gca().set_aspect('equal')
# plt.gca().axis(c.plotbox())
# plt.show()