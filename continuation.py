import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
import shooting
import math

def natural_parameters(f, cmax, cmin, delta_n, u0, ODE):

    n = (cmax - cmin)/delta_n
    k = np.linspace(cmin, cmax, math.floor(abs(n)))
    

    if ODE == False:
        discretisation = lambda u, alpha, f: f(u, alpha)
    else: 
        discretisation = lambda u, alpha, f: shooting.shoot(f, u, alpha)

    sol = []
    sol.append(fsolve(discretisation, u0, args=(cmin, f)))      
    
    for a in range(1, len(k)):
        sol.append(fsolve(discretisation, sol[-1], args=(k[a], f)))   
    

    return sol, k


#def pseudo_arclength():
