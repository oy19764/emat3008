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
from parameter_tests import test_inputs


def solve_pde(u_I, L, T, kappa, pj, qj ,boundary, method, mt=1000, mx=30):
    """ 
    Solves PDE for the given boundary condition under the specified method
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
    # begin input tests
    # test u_I is a function
    test_inputs(u_I, 'u_I', 'test_function')
    # test L is float
    test_inputs(L, 'L', 'test_int_or_float')
    # test T is float
    test_inputs(T, 'T', 'test_int_or_float')
    # test mt is float
    test_inputs(mt, 'mt', 'test_int_or_float')
    # test mx is float
    test_inputs(mx, 'mx', 'test_int_or_float')
    # test kappa is float
    test_inputs(kappa, 'kappa', 'test_int_or_float')
    
    # test string inputs:
    #test boundary conditions input
    if not isinstance(boundary, str):
        raise TypeError(f"Boundary condition specified: {boundary} is not a string.\n"
                        "Please input 'dirichlet' or 'neumann'.")
    else:
        if boundary == 'dirichlet':
            pass
        elif boundary == 'neumann':
            pass
        elif boundary == 'periodic':
            pass
        else:
            raise ValueError(f"Boundary condition specified: {boundary} is not a valid string.\n"
                            "Please input 'dirichlet', 'periodic' or 'neumann'.")
    
    # test solving method input
    if not isinstance(method, str):
        raise TypeError(f"Solving method specified: {method} is not a string.\n"
                        "Please input 'forward', 'backward' or 'crank'.")
    else:
        if method == 'forward':
            pass
        elif method == 'backward':
            pass
        elif method == 'crank':
            pass
        else:
            raise ValueError(f"Solving method specified: {method} is not a valid string.\n"
                            "Please input 'forward', 'backward' or 'crank'.")
    # end input tests


    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    if boundary == 'periodic':
        x = np.linspace(0, L, mx) 
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    #lmbda = kappa*deltax/(deltat**2)
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)


    def boundary_type(boundary):
        """
        Defines the matrix dimensions and initial condition dimensions in function of the boundary
        conditions specified
        """
        if boundary == 'dirichlet':
            matrix_dim = mx - 1
            u_j = np.zeros(x.size)
            for i in range(0,len(u_j)):
                u_j[i] = u_I(x[i])
            
            return matrix_dim, u_j


        if boundary == 'neumann':
            matrix_dim = mx + 1
            u_j = np.zeros(x.size)
            for i in range(0,len(u_j)):
                u_j[i] = u_I(x[i])
            return matrix_dim, u_j

        
        if boundary == 'periodic':
            matrix_dim = mx
            u_j = np.zeros(mx)
            for i in range(0,len(u_j)):
                u_j[i] = u_I(x[i])
            return matrix_dim, u_j


    # define additive vector
    def additive_vector(boundary):
        """ 
        Defines the additive vector size based on the boundary condition being used."""
        if boundary == 'dirichlet':
            return np.zeros(mx-1)
        if boundary == 'neumann':
            return np.zeros(mx+1)
        

    def tri_matrix(method, matrix_dim, boundary):
        """ 
        In-bedded function which returns the appropriate tri-diagonal matrix in function 
        of the boudary condition and methid specified.
        """
        # Set tridiagonal matrix 
        # forward euler
        if method == 'forward':
            diag = [[lmbda] * (matrix_dim-1), [1 - 2*lmbda] * matrix_dim , [lmbda] * (matrix_dim-1)]
            tridiag = diags(diag, offsets = [-1,0,1], format = 'csc')
            if boundary == 'dirichlet':
                return tridiag, None
            elif boundary == 'periodic':
                tridiag = tridiag.toarray()
                tridiag[-1,0] = lmbda
                tridiag[0,-1] = lmbda
                return csr_matrix(tridiag), None
            elif boundary == 'neumann':
                tridiag = tridiag.toarray()
                tridiag[0,1] *= 2
                tridiag[-1,-2] *= 2   
                return csr_matrix(tridiag), None
        
        #backwards euler
        if method == 'backward':
            diag = [[-lmbda] * (matrix_dim-1), [1 + 2*lmbda] * matrix_dim , [-lmbda] * (matrix_dim-1)]
            tridiag = diags(diag, offsets = [-1,0,1], format = 'csc')
            if boundary == 'dirichlet':
                return tridiag, None
            elif boundary == 'periodic':
                tridiag = tridiag.toarray()
                tridiag[-1,0] = lmbda
                tridiag[0,-1] = lmbda
                return csr_matrix(tridiag), None
            elif boundary == 'neumann':
                tridiag = tridiag.toarray()
                tridiag[0,1] = tridiag[0,1]*2
                tridiag[-1,-2] = tridiag[-1,-2]*2
                return csr_matrix(tridiag), None
        
        if method == 'crank':
            diag = [[-lmbda/2] * (matrix_dim-1), [1 + lmbda] * matrix_dim , [-lmbda/2] * (matrix_dim-1)]
            diag2 = [[lmbda/2] * (matrix_dim-1), [1 - lmbda] * matrix_dim , [lmbda/2] * (matrix_dim-1)]
            tridiag = diags(diag, offsets = [-1,0,1], format = 'csc')
            tridiag2 = diags(diag2, offsets = [-1,0,1], format = 'csc')
            if boundary == 'dirichlet':
                return tridiag, tridiag2
            elif boundary == 'periodic':
                tridiag = tridiag.toarray()
                tridiag[-1,0] = lmbda
                tridiag[0,-1] = lmbda
                tridiag2[-1,0] = lmbda
                tridiag2[0,-1] = lmbda
                return csr_matrix(tridiag), csr_matrix(tridiag2)
            elif boundary == 'neumann':
                tridiag = tridiag.toarray()
                tridiag2 = tridiag2.toarray()
                tridiag[0,1] = tridiag[0,1]*2
                tridiag[-1,-2] = tridiag[-1,-2]*2
                tridiag2[0,1] = tridiag2[0,1]*2
                tridiag2[-1,-2] = tridiag2[-1,-2]*2
                return csr_matrix(tridiag), csr_matrix(tridiag2)
                   

    # set up appropriate matrix dimensions
    matrix_dim, u_j = boundary_type(boundary)
    u_jp1 = np.zeros(mx+1)
    
   
    # get diagonals corresponding to the method
    diag1, diag2 = tri_matrix(method, matrix_dim, boundary)


    # setup additive vector
    aV = additive_vector(boundary)
    
    # Solve PDE for each time value
    for i in range(0,mt):
        # forwad euler matrix calc
        if boundary == 'dirichlet':
            aV[0], aV[-1] = pj(t[i]), qj(t[i])
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
            aV[0], aV[-1] = -pj(t[i]), qj(t[i])
            if method == 'forward':
                u_jp1 = diag1.dot(u_j) + 2 * lmbda * deltax * aV
            # backwards euler matrix calc
            if method == 'backward':
                u_jp1 = spsolve(diag1, u_j) + 2 * aV * lmbda
            # crank-nicholson matrix calc
            if method == 'crank': #that soulja boy
                u_jp1 = spsolve(diag1, diag2.dot(u_j)) + 2 * aV * lmbda
                
            u_j[:] = u_jp1[:]

        if boundary == 'periodic':

            if method == 'forward':
                u_jp1 = diag1.dot(u_j)

            if method == 'backward':
                u_jp1 = spsolve(diag1, u_j)

            if method == 'crank':
                u_jp1 = spsolve(diag1, diag2.dot(u_j))

            #u_jp1[0] = p(t[i])

    return x, u_j



if __name__ == '__main__':

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

    # Set numerical parameters
    mx = 30   # number of gridpoints in space
    mt = 1000   # number of gridpoints in time


    # visualising changing the initial conditions

    # def u_exactp(x,t,p):
    #     # the exact solution
    #     y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)**p
    #     return y

    xx = np.linspace(0,L,250)
    # a = np.linspace(0.5,3,6)
    # for n in a:
    #     plt.plot(xx,u_exactp(xx,T,n),label=f'initial condition to the power of {n}')
    # plt.xlabel('x')
    # plt.ylabel('u(x,0.5)')
    # plt.title('Effect of changing the initial conditions to the power of p')
    # plt.legend()
    # plt.show()
    


    # boundary values for dirichlet
    def p(t):
        return 0
    def q(t):
        return 0

    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi*x/L)
        #y = np.sin(pi*x)**0.1
        return y

    def u_exact(x,t):
        # the exact solution
        y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
        return y


    # plot PDE using dirilchet boundary conditions fo the forward and backward euler and
    # crank-nicholson methods. Using boundary conditions = 0

    # Forward Euler
    x, u_jdf = solve_pde(u_I, L, T, kappa, p, q, boundary = 'dirichlet', method = 'forward')
    # Backward Euler
    x, u_jdb = solve_pde(u_I, L, T, kappa, p, q, boundary = 'dirichlet', method = 'backward')
    # Crank-Nicholson
    x, u_jdc = solve_pde(u_I, L, T, kappa, p, q, boundary = 'dirichlet', method = 'crank')
    plt.subplot(1,2,1)
    plt.plot(x,u_jdf, '*', label='Forward Euler')
    plt.plot(x, u_jdb,'*', label='Backward Euler')
    plt.plot(x,u_jdc,'*', label='Crank-Nicholson')
    plt.plot(xx,u_exact(xx,T),label='Exact solution')
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.title('Plot heat pde using dirichlet boundary conditions',fontsize=8)
    plt.legend()
    #plt.show()


    # plot PDE using periodic boundary conditions for the forward euler method. 

    #def p(t):
    #    return 0.01

    # Forward Euler
    x, u_jpf = solve_pde(u_I, L, T, kappa, p, q, boundary = 'periodic', method = 'forward')
    # Backward Euler
    x, u_jpb = solve_pde(u_I, L, T, kappa, p, q, boundary = 'periodic', method = 'backward')
    # Crank-Nicholson
    x, u_jpc = solve_pde(u_I, L, T, kappa, p, q, boundary = 'periodic', method = 'crank')
    plt.subplot(1,2,2)
    plt.plot(x,u_jpf, '*', label='Forward Euler')
    plt.plot(x, u_jpb,'*', label='Backward Euler')
    plt.plot(x,u_jpc,'*', label='Crank-Nicholson')
    plt.plot(xx,u_exact(xx,T),label='Exact solution')
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.title('Plot heat pde using Periodic boundary conditions',fontsize=8)
    plt.legend()
    plt.show()




# plot PDE using neumann boundary conditions for the forward euler method. 
    # Using boundary conditions p = 0, q = 1

    # boundary values for neumann
    def pn(t):
        return 0
    def qn(t):
        return 0.5

    # Forward Euler
    x, u_jnf = solve_pde(u_I, L, T, kappa, pn, qn, boundary = 'neumann', method = 'forward')
    # Backward Euler
    x, u_jnb = solve_pde(u_I, L, T, kappa, pn, qn, boundary = 'neumann', method = 'backward')
    # Crank-Nicholson
    x, u_jnc = solve_pde(u_I, L, T, kappa, pn, qn, boundary = 'neumann', method = 'crank')

    
    plt.plot(x,u_jnf, '*', label='Forward Euler')
    plt.plot(x, u_jnb,'*', label='Backward Euler')
    plt.plot(x,u_jnc,'*', label='Crank-Nicholson')
    #plt.plot(xx,u_exact(xx,T),label='Exact solution')
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.title('Plot heat pde using neumann boundary conditions',fontsize=8)
    plt.legend()
    plt.show()  



    # x, u_jpf = solve_pde(u_I, L, T, kappa, p, q, boundary = 'periodic', method = 'forward')
    # plt.plot(x,u_jpf, '*', label='Forward Euler')
    # plt.title('Plot heat pde using Periodic boundary conditions',fontsize=8)
    # plt.legend()
    # plt.show()