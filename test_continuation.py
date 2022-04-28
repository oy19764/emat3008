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



def natural_parameters(f, cmax, cmin, delta_n, u0, ODE):

    n = (cmax - cmin)/delta_n
    k = np.linspace(cmin, cmax, math.floor(abs(n)))
    

    if ODE == False:
        discretisation = lambda u, alpha, f: f(u, alpha)
    else: 
        discretisation = lambda u, alpha, f: shooting.shoot(f, u, alpha)

    print(u0)
    sol = []
    sol.append(fsolve(discretisation, u0, args=(cmin, f)))      #(lambda U: discretisation(U, cmin, f), u0))
    
    for a in range(1, len(k)):
        sol.append(fsolve(discretisation, sol[-1], args=(k[a], f)))    #lambda U: discretisation(U, k[a], f), sol[-1]))
    

    return sol, k


#sol, alpha = natural_parameters(cubic, cmax, cmin, delta_n, 2, ODE = False)

#plt.plot(alpha, sol)
#plt.show()
#print(sol)


def hopf(u, t, beta):

    u1, u2 = u

    du1dt = beta*u1 - u2 - u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 - u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))

cmin = 0
cmax = 2
delta_n = 0.01 # step size

sol, alpha = natural_parameters(hopf, cmax, cmin, delta_n, (1.4, 0, 6.3), ODE=True)
print(sol)

u1 = [sol[0][0]]
u2 = [sol[0][1]]
for i in range(1,len(alpha)):
    u1.append(sol[i][0])
    u2.append(sol[i][1])
#plt.show()
plt.plot(alpha, u1)
plt.plot(alpha, u2)
plt.show()


def hopf_mod(u, t, beta):

    u1, u2 = u

    du1dt = beta*u1 - u2 + u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2
    du2dt = u1 + beta*u2 + u2*(u1**2 + u2**2) - u2*(u1**2 + u2**2)**2

    return np.array((du1dt, du2dt))