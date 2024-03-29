import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
import math
from tqdm import tqdm
from parameter_tests import test_inputs, test_ode


def natural_parameters(f, par_range, delta_n, u0, discretisation, solver=fsolve, pc=None):
    """
    Natural Parameters continuation method. Finds the solution to the ode function for
    each parameter in the parameter range
        Parameters:
            f (function):               ODE for which numerical continuation is being performed
            par_range (tuple):          parameter range numerical continuation is solving for
            delta_n (float):            increments between parameter values
            u0 (tuple):                 tuple containing estimate for inital values and time period
            discretisation(function):   method
            solver:                     rootfinding method to be used, defaults to fsolve
            pc(function):               phase condition function to be used if solving system of ODE's           
        Output:
            sol(array):                 solution to the function (initial conditions, time period) for each parameter
            alpha(array):               associated parameter value to the solution
    """
    

    # test parameters:
    # test par_range is a tuple
    test_inputs(par_range, 'par_range', 'test_tuple')
    # test delta_n is float
    test_inputs(delta_n, 'delta_n', 'test_int_or_float')
    # test u0 is an integer or tuple
    test_inputs(u0, 'u0', 'test_int_or_tuple')
    # test discretisation is a function
    test_inputs(discretisation, 'discretisation', 'test_function')
    # test f is a valid function
    # if isinstance(u0, tuple):
    #     test_ode(f, u0[:-1], par_range[0])
    # else:
    #     test_ode(f,u0, par_range[0])
    

    param_n = (par_range[1] - par_range[0])/delta_n     # number of parameters
    param_list = np.linspace(par_range[0], par_range[1], math.floor(abs(param_n)))  #parameter list

    def get_sol(f, alpha, discretisation, u0, pc):

        if pc == None:
            return solver(discretisation(f), u0, args=alpha)

        else:
            return solver(discretisation(f), u0, args=(pc, alpha))


    # get dimensions of solution vector, and generate solution array
    if pc == None:
        sol = np.zeros(len(param_list))
    else:
        n = len(u0)
        sol = np.zeros([len(param_list), n])
    
    # get first estimate
    sol[0] = get_sol(f, param_list[0], discretisation, u0, pc)
    print(sol[0])
    for a in tqdm(range(1, len(param_list))):
        sol[a]= get_sol(f, param_list[a], discretisation, np.round(sol[a-1], 3), pc)
          
    return sol, param_list




# unsuccessful attempt at pseudo arclength continuation
def pseudo_arclength(f, par_range, delta_n, u0, discretisation, solver=fsolve, pc=None):



    # test parameters:
    # test par_range is a tuple
    test_inputs(par_range, 'par_range', 'test_tuple')
    # test delta_n is float
    test_inputs(delta_n, 'delta_n', 'test_int_or_float')
    # test u0 is a tuple
    test_inputs(u0, 'u0', 'test_tuple')
    # test discretisation is a function
    test_inputs(discretisation, 'discretisation', 'test_function')
    # test f is a valid function
    test_ode(f, u0[:-1], par_range[0])
    

    param_n = (par_range[1] - par_range[0])/delta_n     # number of parameters
    param_list = np.linspace(par_range[0], par_range[1], math.floor(abs(param_n)))  #parameter list
    
    
    def get_sol(f, alpha, discretisation, u0, pc):

        if pc == None:
            return solver(discretisation(f), u0, args=alpha)
        else:
            return(discretisation(f, u0, pc, alpha))


    # two input parameters: v0 and v1  -  v0 = [alpha0, u0], v1 = [alpha1, u1]
    
    """
    given (u0,h0)
    calculate Gu, Gh in (u0,h0)
    calculate du0, dh0
    initial guess: 
        ug = u0 + du0*ds
        hg = h0 + dh0*ds
    use (ug,hg) to solve:
        G(u1,h1) = 0
        (u1-u0)*du0 + (h1-h0)*dh0 - ds = 0
    return (u1,h1)
    given (u1,h1)
    ...
    """


    def get_vectors(u0):

        diff = 0.01 * np.sign(param_list[1]-param_list[0])
        alpha0 = param_list[0]
        alpha1 = alpha0 + diff

        v0 = get_sol(f, alpha0, discretisation, u0, pc)     # v0 = [alpha0, u0]
        v1 = get_sol(f, alpha1, discretisation, v0, pc)     # v1 = [alpha1, u1]

        return v0, v1, alpha0, alpha1



    # generate secant: change = vi - vi-1
    v0, v1, alpha0, alpha1 = get_vectors(u0)

    deltaV = v1 - v0
    deltaA = alpha1 - alpha0

    # predict solution: v~i+1 = vi +change (approximation)
    predV = v1 + deltaV
    predA = alpha1 + deltaA

    pred_vec = np. append(predV, predA)

    # solve augmented root finding problem: (vi+1 - v~i+1)*change = 0
    #sol = 

    #def aug_root_finder(f, pred_vec, v1, discretisation, deltaV, deltaA):


    # repeat until all parameters covered


    return param_n, param_list



if __name__ == '__main__':

    from shooting import shoot
    import matplotlib.pyplot as plt


    def cubic(x, c):

        return x**3 - x + c

    cmax = -2
    cmin = 2
    delta_n = 0.001 # step size
    pararmeter_range = (-2,2)
    discretisation = lambda x: x
    sol, alpha = natural_parameters(cubic, pararmeter_range, delta_n, 2, discretisation, solver=fsolve)
    plt.subplot(1,2,1)
    plt.plot(alpha, sol)
    plt.title('Natural Parameter continuation for a cubic equation')
    plt.xlabel('parameter value')
    plt.ylabel('solution')
    #plt.show()



    def hopf(t, u, beta):

        u1, u2 = u

        du1dt = beta*u1 - u2 - u1*(u1**2 + u2**2)
        du2dt = u1 + beta*u2 - u2*(u1**2 + u2**2)

        return np.array((du1dt, du2dt))


    delta_n = 0.01 # step size
    pararmeter_range = (2,-1)   # changed parameter range to avoid code crashing
    discretisation = shoot
    def pc(u0, *args):
        return hopf(0, u0, *args)[0]

    U0 = (1.2,1.2,6.4)
    sol, alpha = natural_parameters(hopf, pararmeter_range, delta_n, U0, discretisation, fsolve, pc)

    
    u1 = sol[:,0]
    plt.subplot(1,2,2)
    plt.plot(alpha, u1)
    plt.title('Natural Parameter continuation for hopf equation')
    plt.xlabel('parameter value')
    plt.ylabel('solution')
    plt.show()



    # plot modified hopf
    def hopf_mod(t, u, beta):

        u1, u2 = u

        du1dt = beta*u1 - u2 + u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2
        du2dt = u1 + beta*u2 + u2*(u1**2 + u2**2) - u2*(u1**2 + u2**2)**2

        return np.array((du1dt, du2dt))


    delta_n = 0.01 # step size
    pararmeter_range = (2,-0.5) # changed parameter range to avoid code crashing
    discretisation = shoot
    def pc(u0, *args):
        return hopf_mod(0, u0, *args)[0]

    U0 = (1.2,1.2,6.4)
    sol, alpha = natural_parameters(hopf_mod, pararmeter_range, delta_n, U0, discretisation, fsolve, pc)

    
    u1 = sol[:,0]
    plt.plot(alpha, u1)
    plt.title('Natural Parameter continuation for modified hopf equation')
    plt.xlabel('parameter value')
    plt.ylabel('solution')
    plt.show()