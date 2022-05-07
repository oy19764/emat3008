import numpy as np
import matplotlib.pyplot as plt
from math import pi
from ODE_solver import test_inputs
from pde import solve_pde



# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0


# Set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5         # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    #y = np.sin(pi*x)**0.1
    return y


def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# Set numerical parameters
mx = 10    # number of gridpoints in space
mt = 1000   # number of gridpoints in time

# boundary values
def p(t):
    return 0
def q(t):
    return 0


x, u_j = solve_pde(u_I, L, T, mt, mx, kappa, p, q, boundary = 'dirichlet', method = 'crank')

print(u_j)
# plot the final result and exact solution
plt.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend()
plt.show()


p = np.linspace(0,3,7)
x1 = np.linspace(0,10,11)
for n in p:



    
    x, u_j = solve_pde(u_I, L, n, mt, mx, kappa, p, q, boundary = 'dirichlet', method = 'forward')
    plt.plot(x1, u_j, label=f'PDE solution for T={n}')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()  
