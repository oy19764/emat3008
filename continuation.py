import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
import math
from tqdm import tqdm
from ODE_solver import test_inputs

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
            pc(function):               phase condition function to be used if solving system of ODE's           """
    

    # test parameters:
    # test par_range is a tuple
    test_inputs(par_range, 'par_range', 'test_tuple')
    # test delta_n is float
    test_inputs(delta_n, 'delta_n', 'test_int_or_float')
    # test u0 is a tuple
    test_inputs(u0, 'u0', 'test_array')
    # test discretisation is a function
    test_inputs(discretisation, 'discretisation', 'test_function')
    # test f is a function
    if callable(f):
        # test f returns valid output
        t_test = u0[-1]
        test_output = f(u0[:-1], t_test, par_range[0])
        # test valid output type
        if isinstance(test_output, (int, float, np.int_, np.float_, list, np.ndarray)):
            # test valid output size
            if np.array(test_output).shape == np.array(u0[:-1]).shape:
               pass
            else:
                raise ValueError(f"Invalid output shape from ODE function {f},\n"
                                "u0[:-1] and {f} output should have the same shape")
        else:
            raise TypeError(f"Output from ODE function is a {type(test_output)},\n"
            "output should be an integer, float, list or array")
    else: 
        raise TypeError(f"{f} is not a valid input type. \n" 
                            "Please input a function")
    

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
