import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
import shooting
import math
from tqdm import tqdm

def natural_parameters(f, cmax, cmin, delta_n, u0, discretisation, solver=fsolve, pc=None):


    def get_sol(f, alpha, discretisation, u0, pc):

        if pc == None:
            #arg = alpha
            return solver(discretisation(f), u0, args=alpha)
        else:
            #arg = (pc, alpha)
            return(solver(lambda U: discretisation(f, U, pc, alpha), np.round(u0, 5)))


        #return solver(discretisation(f), u0, args=arg)

        # fsolve(lambda U: shoot(ODE, U, pc, *args), U0)

    param_n = (cmax - cmin)/delta_n     # number of parameters
    param_list = np.linspace(cmin, cmax, math.floor(abs(param_n)))  #parameter list
    

    sol = []  
    # if ODE == False:   
    #     sol.append(fsolve(lambda u: f(u, cmin), u0)) # discretisation for non-ODE
    # else:
    #     sol.append(shooting.limit_cycle(f, u0, pc, cmin)) # discretisation for ODE


    sol.append(get_sol(f, cmin, discretisation, u0, pc))

    for a in tqdm(range(1, len(param_list))):

        sol.append(get_sol(f, param_list[a], discretisation, sol[-1], pc))

        # if ODE == False:   
        #     sol.append(fsolve(lambda u: f(u, param_list[a]), sol[-1]))
        # else:
        #     sol.append(shooting.limit_cycle(f, sol[-1], pc, param_list[a]))
          
    return sol, param_list


def pseudo_arclength(f, cmax, cmin, delta_n, u0, ODE, pc):

    if ODE == False:
        discretisation = lambda u, alpha, f: f(u, alpha)
    else: 
        discretisation = lambda u, alpha, f: shooting.shoot(f, u, alpha, pc)

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
