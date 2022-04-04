import ODE_solver
import math
import numpy as np
import matplotlib.pyplot as plt


""" initial conditions """
X0 = [1]
t0 = 0
h = 0.2
deltat_max = 1
odearray = np.linspace(0,1,4)

'initial vector conditions'
x0 = 0
a0 = 1
X0 = [x0, a0]


def f(t, x):
    
    return x



"""
x'' = -x
equivalent to:
x' = a
a' = -x
treating as a vector:
(x, a)' = (a, -x)
make function that can compute the rhs of the vector:
X' = f(X, t)


def f(t, X):
    #print(X)
    #print(type(X))
    x = X[0]
    a = X[1]
    dxdt = a
    dadt = -x
    dXdt = [dxdt, dadt]
    
    return dXdt
"""







x = ODE_solver.solve_ode(f, odearray, X0)
