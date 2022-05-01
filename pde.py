# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve_pde(u_I, boundary,L, T, mt, mx, kappa, method):
    """ 
    Makes a single Euler step of step size h for the given function.
        Parameters:
            u_I(function)       initial condition equation, takes x as arguments
            boundary(function)  boundary conditions, takes x, t as arguments
            L(float):           length of spatial domain
            T(float):           total time to solve for
            mt(int):            number of gridpoints in time
            mx(int):            number of gridpoints in space
            kappa(float):       diffusion constant
        Returns:
    """
    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)

    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])
        

    # Set tridiagonal matrix 
    # forward euler
    if method == 'forward':
        diag = [[lmbda] * (mx-1), [1 - 2*lmbda] * mx , [lmbda] * (mx-1)]
    
    #backwards euler
    if method == 'backward':
        diag = [[-lmbda] * (mx-1), [1 + 2*lmbda] * mx , [-lmbda] * (mx-1)]
    
    if method == 'crank':
        diag = [[-lmbda/2] * (mx-1), [1 + lmbda] * mx , [-lmbda/2] * (mx-1)]
        diag2 = [[lmbda/2] * (mx-1), [1 - lmbda] * mx , [lmbda/2] * (mx-1)]
    
    tridiag = diags(diag, offsets = [-1,0,1], format = 'csc')
    tridiag2 = diags(diag2, offsets = [-1,0,1], format = 'csc')

    

    for i in range(0,mt):
        # forwad euler matrix calc
        if method == 'forward':
            u_jp1[1:] = tridiag.dot(u_j[1:])

        # backwards euler matrix calc
        if method == 'backward':
            u_jp1[1:] = spsolve(tridiag, u_j[1:])

        # crank-nicholson matrix calc
        if method == 'crank':
            u_jp1[1:] = spsolve(tridiag, tridiag2.dot(u_j[1:]))

        # boundary conditions
        u_jp1[0] = boundary(0, t[i])
        u_jp1[mx] = boundary(L, t[i])
        #initialise u_j for the next time step
        u_j[:] = u_jp1[:]

    return x, u_j






# Set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5         # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    #y = np.sin(pi*x)**0.1
    return y

def heat_boundary(x,t):
    # Boundary conditions u(0,t) = u(L,t) = 0
    return 0


def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# Set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000   # number of gridpoints in time


x, u_j = solve_pde(u_I, heat_boundary, L, T, mt, mx, kappa, method = 'crank')

# Plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend()
pl.show()