#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy
from scipy import signal
from scipy.integrate import quad

w = 1
T_wdt = (2 * numpy.pi) / w
T = (0, T_wdt)
N = list(range(-50, 51))

func = signal.sawtooth

x = numpy.arange(-T_wdt * 2, T_wdt * 2, 0.1)
y = [func(t) for t in x]

def complex_quadrature(func, a, b):
    def real_func(x):
        return numpy.real(func(x))
    def imag_func(x):
        return numpy.imag(func(x))
    
    real_integral = quad(real_func, a, b)
    imag_integral = quad(imag_func, a, b)

    return real_integral[0] + 1j*imag_integral[0]

coeff = [complex_quadrature(lambda t: func(t) * numpy.exp(- 1j * n * w * t), T[0], T[1]) / T_wdt for n in N]
rstrd = [sum([coeff[n + numpy.abs(N[0])] * numpy.exp(1j * w * n * t) for n in N]) for t in x]

mods = [numpy.abs(e) for e in coeff]
args = [numpy.angle(e) for e in coeff]

plt.subplot(3, 1, 1)
plt.title("Spettro di ampiezza")
plt.ylabel("module")
plt.xlabel("freq")
plt.plot(N, mods, '.', color='black')
plt.vlines(N, ymin=0, ymax=[ a if a > 0 else 0 for a in mods ], colors='black', linestyles='solid')
plt.axhline(y=0, color='black')

plt.subplot(3, 1, 2)
plt.title("Spettro di fase")
plt.ylabel("arg")
plt.xlabel("freq")
plt.plot(N, args, '.', color='black')
plt.vlines(N, ymin=[ a if a < 0 else 0 for a in args ], ymax=[ a if a > 0 else 0 for a in args ], colors='black', linestyles='solid')
plt.axhline(y=0, color='black')

plt.subplot(3, 1, 3)
plt.title("Funzione")
plt.plot(x, y)
plt.plot(x, rstrd, color='red')
plt.ylabel("value")
plt.xlabel("time")

plt.tight_layout()
plt.show()
