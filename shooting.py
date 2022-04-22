import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.optimize import fsolve




def G(ODE, u0, T, *args):

            sol = os.solve_ode(ODE, os.rk4_step, [0, T], u0, 0.001, *args)
            g1 = u0 - sol[-1]
            g2 = ODE(u0, 0,*args)[1] # phase condition
            gsol = np.append(g1, g2)
            return gsol


def shoot(ODE, U, *args):

        u0 = U[:-1]
        T = U[-1]
        shot = G(ODE, u0, T, *args)

        return shot


def limit_cycle(ODE ,u0, T, *args):


    sol = fsolve(lambda U: shoot(ODE, U, *args), np.concatenate((u0, [T])))
    u0 = sol[:-1]
    T = sol[-1]
    return u0, T