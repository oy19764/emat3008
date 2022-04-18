import math
import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
"""
def f(x, t):
    
    return x
"""




def f(X, t, a=1, b=0.30, d=0.1):

    x = X[0]
    y = X[1]

    dxdt = x*(1-x)-((a*x*y)/(d+x))
    dydt = b*y*(1-(y/x))
    
    return np.array((dxdt, dydt))

x0 = [10,2]
t = np.linspace(0,100,10)
h = 0.001



a = os.solve_ode(f, os.rk4_step, t, x0, h)
print(a)
