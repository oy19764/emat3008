import numpy as np
import ODE_solver as os
from scipy import optimize


def G(f, u0, t, *args):
    sol = os.solve_ode(f, os.rk4_step, t, u0, 0.001, *args)
    return u0 - sol





def limit_cycle(f, u0, t0_guess, T_guess, *args):
    
    sol = optimize.root
