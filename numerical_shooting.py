import math
import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.integrate import odeint

## Exercise 1  ##

# initial conditions #


def fpred(X, t, a=1, b=0.30, d=0.1):

    print(X)
    print(type(X))
    x = X[0]
    y = X[1]
    #x, y = v
    #x = v[0]
    #y = v[1]

    dxdt = x*(1-x)-(a*x*y)/(d+x)
    dydt = b*y*(1-(y/x))
    print([dxdt, dydt])
    return np.array((dxdt, dydt))

# initial conditions #
t = np.linspace(0,1,10)
x = 10
y = 2
v0 = [x, y]
h = 0.01
deltat_max = 1
a=1
b=0.20
d=0.1



#a = os.solve_ode(fpred, os.euler_step, t, v0, h)
a = odeint(fpred, v0, t, args=(a,b,d))
print(a)
