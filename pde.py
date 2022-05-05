# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from ODE_solver import test_inputs


def solve_pde(u_I, L, T, mt, mx, kappa, pj, qj ,boundary, method):
    """ 
    Makes a single Euler step of step size h for the given function.
        Parameters:
            u_I(function)       initial condition equation, takes x as arguments
            L(float):           length of spatial domain
            T(float):           total time to solve for
            mt(int):            number of gridpoints in time
            mx(int):            number of gridpoints in space
            kappa(float):       diffusion constant
            pj(int):            left boundary value
            qj(int):            right boundary value
            boundary(str)       boundary conditions
            method(str):        method to use to solve PDE
        Returns:
    """
    # 

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)


    def boundary_type(boundary):
        if boundary == 'dirilichet':
            matrix_dim = mx - 1
            print(matrix_dim)
            u_j = np.zeros(x.size)
            #u_j[0] = pj
            #u_j[-1] = qj
            for i in range(0,len(u_j)):
                u_j[i] = u_I(x[i])

            return matrix_dim, u_j


        if boundary == 'neumann':
            matrix_dim = mx + 1
            u_j = np.zeros(x.size)
            
            for i in range(0,len(u_j)):
                u_j[i] = u_I(x[i])
            return matrix_dim, u_j

    # define additive vector
    def additive_vector(boundary):
        if boundary == 'dirilichet':
            return np.zeros(mx-1)
        if boundary == 'neumann':
            return np.zeros(mx+1)
        

    def tri_matrix(method, matrix_dim, boundary):

        # Set tridiagonal matrix 
        # forward euler
        if method == 'forward':
            diag = [[lmbda] * (matrix_dim-1), [1 - 2*lmbda] * matrix_dim , [lmbda] * (matrix_dim-1)]
            tridiag = diags(diag, offsets = [-1,0,1], format = 'csc')
            if boundary == 'dirilichet':
                return tridiag, None
            if boundary == 'neumann':
                tridiag = tridiag.toarray()
                tridiag[0,1] *= 2
                tridiag[-1,-2] *= 2   
                return csr_matrix(tridiag), None
        
        #backwards euler
        if method == 'backward':
            diag = [[-lmbda] * (matrix_dim-1), [1 + 2*lmbda] * matrix_dim , [-lmbda] * (matrix_dim-1)]
            tridiag = diags(diag, offsets = [-1,0,1], format = 'csc')
            if boundary == 'dirilichet':
                return tridiag, None
            if boundary == 'neumann':
                tridiag = tridiag.toarray()
                tridiag[0,1] = tridiag[0,1]*2
                tridiag[-1,-2] = tridiag[-1,-2]*2
                return csr_matrix(tridiag), None
        
        if method == 'crank':
            diag = [[-lmbda/2] * (matrix_dim-1), [1 + lmbda] * matrix_dim , [-lmbda/2] * (matrix_dim-1)]
            diag2 = [[lmbda/2] * (matrix_dim-1), [1 - lmbda] * matrix_dim , [lmbda/2] * (matrix_dim-1)]
            tridiag = diags(diag, offsets = [-1,0,1], format = 'csc')
            tridiag2 = diags(diag2, offsets = [-1,0,1], format = 'csc')
            if boundary == 'dirilichet':
                return tridiag, tridiag2
            if boundary == 'neumann':
                tridiag = tridiag.toarray()
                tridiag2 = tridiag2.toarray()
                tridiag[0,1] = tridiag[0,1]*2
                tridiag[-1,-2] = tridiag[-1,-2]*2
                tridiag2[0,1] = tridiag2[0,1]*2
                tridiag2[-1,-2] = tridiag2[-1,-2]*2
                return csr_matrix(tridiag), csr_matrix(tridiag2)
            
                

    matrix_dim, u_j = boundary_type(boundary, pj, qj)
    u_jp1 = np.zeros(mx+1)
    
   
    # get diagonals corresponding to the method
    diag1, diag2 = tri_matrix(method, matrix_dim, boundary)


    # setup additive vector
    aV = additive_vector(boundary)
    

    for i in range(0,mt):
        # forwad euler matrix calc
        if boundary == 'dirilichet':
            aV[0], aV[-1] = pj, qj
            if method == 'forward':
                u_jp1[1:-1] = diag1.dot(u_j[1:-1]) + aV * lmbda

            # backwards euler matrix calc
            if method == 'backward':
                u_jp1[1:-1] = spsolve(diag1, u_j[1:-1]) + aV * lmbda

            # crank-nicholson matrix calc
            if method == 'crank': #that soulja boy
                u_jp1[1:-1] = spsolve(diag1, diag2.dot(u_j[1:-1])) + aV * lmbda

            # add boundary conditions
            u_jp1[0] = aV[0]
            u_jp1[-1] = aV[-1]
            #initialise u_j for the next time step
            u_j[:] = u_jp1[:]

        if boundary == 'neumann':
            aV[0], aV[-1] = -pj, qj
            if method == 'forward':
                u_jp1 = diag1.dot(u_j) + 2 * lmbda * deltax * aV
            # backwards euler matrix calc
            if method == 'backward':
                u_jp1 = spsolve(diag1, u_j) + aV * lmbda
            # crank-nicholson matrix calc
            if method == 'crank': #that soulja boy
                u_jp1 = spsolve(diag1, diag2.dot(u_j)) + aV * lmbda
                
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
mx = 10    # number of gridpoints in space
mt = 1000   # number of gridpoints in time

# boundary values
p = 0
q = 0


x, u_j = solve_pde(u_I, L, T, mt, mx, kappa, p, q, boundary = 'neumann', method = 'crank')

# plot the final result and exact solution
plt.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
plt.plot(xx,u_exact(xx,T),'b-',label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend()
plt.show()
