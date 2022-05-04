import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.optimize import fsolve
from continuation import natural_parameters, pseudo_arclength
from shooting import shoot

def cubic(x, c):

    return x**3 - x + c

cmax = 2
cmin = -2
delta_n = 0.001 # step size

discretisation = lambda x: x
sol, alpha = natural_parameters(cubic, cmax, cmin, delta_n, 2, discretisation, solver=fsolve)
plt.plot(alpha, sol)
plt.show()



def hopf(u, t, beta):

    u1, u2 = u

    du1dt = beta*u1 - u2 - u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 - u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))

cmin = 0
cmax = 2
delta_n = 0.01 # step size

discretisation = shoot
sol, alpha = natural_parameters(hopf, cmax, cmin, delta_n, (1.2, 1.2, 6.3), discretisation, solver=fsolve, pc=1)
print(sol)

u1 = [sol[0][0]]
u2 = [sol[0][1]]
#T = [sol[0][2]]
for i in range(1,len(alpha)):
    u1.append(sol[i][0])
    u2.append(sol[i][1])
#    T.append(sol[i][2])

plt.plot(alpha, u1)
plt.plot(alpha, u2)
plt.show()

def true_hopf(t, beta, theta = 0.0):

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)

    return np.array([u1, u2])




#true_roots = []
#for a, t in list(zip(alpha, T)):
#    
#    true_roots.append(true_hopf(0, a, theta = t))#

#print(true_roots)
#root_u1 = []
#root_u2 = []
#for i in range(0,len(alpha)):
#    root_u1.append(true_roots[i][0])
#    root_u2.append(true_roots[i][1])

#plt.plot(alpha, root_u1)
#plt.plot(alpha, root_u2)
#plt.show()


def hopf_mod(u, t, beta):

    u1, u2 = u

    du1dt = beta*u1 - u2 + u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2
    du2dt = u1 + beta*u2 + u2*(u1**2 + u2**2) - u2*(u1**2 + u2**2)**2

    return np.array((du1dt, du2dt))


#print(pseudo_arclength(hopf, cmax, cmin, delta_n, (1.4,0,6), ODE = True))
