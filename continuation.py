import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
import shooting
import math

def natural_parameters(f, cmax, cmin, delta_n, u0, ODE, pc):

    n = (cmax - cmin)/delta_n
    k = np.linspace(cmin, cmax, math.floor(abs(n)))
    

    sol = []  
    if ODE == False:   
        sol.append(fsolve(lambda u: f(u, cmin), u0))
    else:
        sol.append(shooting.limit_cycle(f, u0, pc, cmin))

    for a in range(1, len(k)):

        if ODE == False:   
            sol.append(fsolve(lambda u: f(u, k[a]), sol[-1]))
        else:
            sol.append(shooting.limit_cycle(f, sol[-1], pc, k[a]))
          
    
    return sol, k


def pseudo_arclength(f, cmax, cmin, delta_n, u0, ODE, pc):

    if ODE == False:
        discretisation = lambda u, alpha, f: f(u, alpha)
    else: 
        discretisation = lambda u, alpha, f: shooting.shoot(f, u, alpha, pc)

    # two input parameters: v0 and v1  -  v0 = [alpha0, u0], v1 = [alpha1, u1]
    n = (cmax - cmin)/delta_n
    k = np.linspace(cmin, cmax, math.floor(abs(n)))
    diff = 0.01 * np.sign(cmax-cmin)
    alpha0 = k[0]
    alpha1 = alpha0 + diff
    v0 = np.append(fsolve(discretisation, u0, args=(alpha0, f)))        # v0 = [alpha0, u0]
    v1 = np.append(fsolve(discretisation, v0[:-1], args=(alpha1, f)))   # v1 = [alpha1, u1]

    # generate secant: change = vi - vi-1



    # predict solution: v~i+1 = vi +change (approximation)

    # solve augmented root finding problem: (vi+1 - v~i+1)*change = 0

    # repeat until all parameters covered


    return n, k
