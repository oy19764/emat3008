import math
import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os


## Exercise 1  ##

# initial conditions #
t = np.linspace(0,100,101)
x = 100
y = 100
v = [x, y]
h = 0.0001
deltat_max = 1



def fpred(t, v, t, a=1, b=0.25, d=0.1):

    x = v[0]
    y = v[1]

    dxdt = x*(1-x)-(a*x*y)/(d+x)
    dydt = b*y*(1-y/x)

    return [dxdt, dydt]



a = os.solve_ode(fpred, os.rk4_step, t, v, h, deltat_max)
