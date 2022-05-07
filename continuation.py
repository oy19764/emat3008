import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
import math
from tqdm import tqdm

def natural_parameters(f, par_range, delta_n, u0, discretisation, solver=fsolve, pc=None):
    """
    Natural Parameters continuation method. Finds the solution to the ode function for
    each parameter in the parameter range"""
    

    param_n = (par_range[1] - par_range[0])/delta_n     # number of parameters
    param_list = np.linspace(par_range[0], par_range[1], math.floor(abs(param_n)))  #parameter list

    def get_sol(f, alpha, discretisation, u0, pc):

        if pc == None:
            return solver(discretisation(f), u0, args=alpha)
        else:
            return solver(discretisation(f), u0, args=(pc, alpha))


    sol = []  
    sol.append(get_sol(f, param_list[0], discretisation, u0, pc))

    for a in tqdm(range(1, len(param_list))):
        sol.append(get_sol(f, param_list[a], discretisation, np.round(tuple(sol[-1]), 3), pc))

          
    return sol, param_list





def pseudo_arclength(f, cmax, cmin, delta_n, u0, ODE, discretisation, solver=fsolve, pc=None):

    def get_sol(f, alpha, discretisation, u0, pc):

        if pc == None:
            return solver(discretisation(f), u0, args=alpha)
        else:
            return(discretisation(f, u0, pc, alpha))


    # two input parameters: v0 and v1  -  v0 = [alpha0, u0], v1 = [alpha1, u1]
    param_n = (cmax - cmin)/delta_n
    param_list = np.linspace(cmin, cmax, math.floor(abs(param_n)))
    diff = 0.01 * np.sign(cmax-cmin)
    alpha0 = param_list[0]
    alpha1 = alpha0 + diff
    v0 = np.append(fsolve(discretisation, u0, args=(alpha0, f)))        # v0 = [alpha0, u0]
    v1 = np.append(fsolve(discretisation, v0[:-1], args=(alpha1, f)))   # v1 = [alpha1, u1]

    # generate secant: change = vi - vi-1



    # predict solution: v~i+1 = vi +change (approximation)

    # solve augmented root finding problem: (vi+1 - v~i+1)*change = 0

    # repeat until all parameters covered


    return param_n, param_list
