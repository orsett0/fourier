#!/usr/bin/env python3
#
# Copyright (C) 2023 Alessio Orsini <alessiorsini.ao@proton.me>
# SPDX-License-Identifier: GPL-3.0-or-later
# 
# A copy of the GNU General Public License is available in the 
# LICENSE file, in the project root directory.

import matplotlib.pyplot as plt
import numpy
from scipy import signal, integrate

w = numpy.pi / 4
T_wdt = (2 * numpy.pi) / w
T = (0, T_wdt)
duty = 0.8
th = (T_wdt * duty) / 2

resolution = 100
N = list(range(-resolution // 2, resolution // 2 + 1))

def func(t):
    t_adj = t + th
    is_duty = t_adj - T_wdt * (t_adj // T_wdt) <= T_wdt * duty

    return 1 if is_duty else 0

print(f"{w=}, {T_wdt=}, {duty=}, {th=}")

print(f"calculating real function...")
x = numpy.arange(-T_wdt * 2, T_wdt * 2, 0.1)
y = [func(t) for t in x]

# This takes a lot of time. Maybe i can find a way to speed things up a bit.
# Compute the integral splitting a complex number by it's real and imaginary parts
def complex_quadrature(func, a, b):
    def real_func(x):
        return numpy.real(func(x))
    def imag_func(x):
        return numpy.imag(func(x))
    
    real_integral = integrate.quad(real_func, a, b, limit=resolution)
    imag_integral = integrate.quad(imag_func, a, b, limit=resolution)

    return real_integral[0] + 1j*imag_integral[0]

print(f"calculating coefficients...")
coeff = [complex_quadrature(lambda t: func(t) * numpy.exp(- 1j * n * w * t), T[0], T[1]) / T_wdt for n in N]
mods = [numpy.abs(e) for e in coeff]
args = [numpy.angle(e) for e in coeff]

print(f"calculating restored function...")
rstrd = [sum([coeff[n + numpy.abs(N[0])] * numpy.exp(1j * w * n * t) for n in N]) for t in x]

plt.subplot(3, 1, 1)
plt.title("Spettro di ampiezza")
plt.ylabel("module")
plt.xlabel("freq")
plt.vlines(N, ymin=0, ymax=[ a if a > 0 else 0 for a in mods ], colors='grey', linestyles='solid')
plt.plot(N, mods, '.', color='black')
plt.axhline(y=0, color='black')

plt.subplot(3, 1, 2)
plt.title("Spettro di fase")
plt.ylabel("arg")
plt.xlabel("freq")
plt.vlines(N, ymin=[ a if a < 0 else 0 for a in args ], ymax=[ a if a > 0 else 0 for a in args ], colors='grey', linestyles='solid')
plt.plot(N, args, '.', color='black')
plt.axhline(y=0, color='black')

plt.subplot(3, 1, 3)
plt.title("Funzione")
plt.plot(x, y)
plt.plot(x, rstrd, color='red')
plt.ylabel("value")
plt.xlabel("time")

plt.tight_layout()
plt.show()
