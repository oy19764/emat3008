import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.integrate import odeint
from scipy import optimize

## Exercise 1  ##

def f(X, t, a=1, b=0.20, d=0.1):

    x = X[0]
    y = X[1]

    dxdt = x*(1-x)-((a*x*y)/(d+x))
    dydt = b*y*(1-(y/x))
    
    return np.array((dxdt, dydt))



# initial conditions
x0 = [3,2]
t = np.linspace(0,100,1001)
h = 0.001



sol = os.solve_ode(f, os.rk4_step, t, x0, h)
#sol = odeint(fpred, v0, t, args=(a,b,d))
#print(sol)

# plot time series

x = sol[:,0]

y = sol[:,1]


plt.plot(t, x)
plt.plot(t, y)
#plt.show()

"""
    The populations flattens out when b>0.26, and it will continue
continue oscillating if b<0.26
"""

## Isolate a periodic orbit ##

# Used to find a good starting point when shooting
#for i in range(len(t)):
#    if abs(x[i] - y[i]) <= 0.001:
#        print(x[i], y[i])
#        print(t[i])

#results: x = 0.14973664465794942  y = 0.15001583632703638 t = 51.300000000000004
# initial conditions:
u0 = [0.15, 0.15]     # using same known initial conitions
t0 = 51.3      # estimate of the time period
t = np.linspace(t0,100,1001)



def G(f, u0, t, *args):

    sol = os.solve_ode(f, os.rk4_step, t, u0, 0.001, *args)
    return u0 - sol

sol2 = G(f, u0, t)
print(sol2)
