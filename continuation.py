import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
import ODE_solver as os
from scipy.optimize import fsolve
from scipy.integrate import odeint
import shooting
import math

def cubic(x, c):

    return x**3 - x + c

cmax = 2
cmin = -2
delta_n = 0.001 # step size



def natural_parameters(f, cmax, cmin, delta_n, u0):

    n = (cmax - cmin)/delta_n
    k = np.linspace(cmin, cmax, math.floor(abs(n)))

    discretisation = lambda u, alpha, f: f(u, alpha)


    sol = []
    sol.append(fsolve(lambda U: discretisation(U, cmin, f), u0))

    for a in range(1, len(k)):

        
        sol.append(fsolve(lambda U: discretisation(U, k[a], f), sol[-1]))
    sol = np.concatenate(sol)
    return sol, k

"""
sol, alpha = natural_parameters(cubic, cmax, cmin, delta_n, 2)

plt.plot(alpha, sol)
plt.show()
"""


def hopf(u, beta):

    u1, u2 = u

    du1dt = beta*u1 - u2 - u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 - u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))

cmin = 0
cmax = 2
delta_n = 0.2 # step size

sol, alpha = natural_parameters(hopf, cmax, cmin, delta_n, (0.5,0.4))
print(sol)

#plt.plot(alpha, sol[0,:])
#plt.plot(alpha,sol[1,:])
#plt.show()








def hopf_mod(u, t, beta):

    u1, u2 = u

    du1dt = beta*u1 - u2 + u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2
    du2dt = u1 + beta*u2 + u2*(u1**2 + u2**2) - u2*(u1**2 + u2**2)**2

    return np.array((du1dt, du2dt))