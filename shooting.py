import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve



def G(ODE, u0, T, *args):
    """
    Returns the difference between the initial conditions and the solution to the ODE,
    and finds the phase condition.
        Parameters:
                ODE (function):         The ODE of which the limit cycle is to be found
                u0 (tuple):             Estimate of the startig conditions
                T (float):              Esstimate of the time period
                args:                   Any additional arguments required for the ODE
        """    

    sol = os.solve_ode(ODE, os.rk4_step, [0, T], u0, 0.001, *args)
    g1 = u0 - sol[-1]
    g2 = ODE(u0, 0,*args)[0] # phase condition
    gsol = np.append(g1, g2)
    return gsol


def shoot(ODE, U, *args):

    u0 = U[:-1]
    T = U[-1]
    shot = G(ODE, u0, T, *args)

    return shot




def limit_cycle(ODE ,U0, *args):
    """
    Solves the shooting problem to find the roots to G, hence u0 ant T which are
    the conditions to make a limit cycle
        Parameters:
                ODE (function):         The ODE of which the limit cycle is to be found
                U0 (tuple):             The estimated initial conditions and time period of the ODE limit cycle
                args:                   Any additional arguments required for the ODE
                
        Returns:
                u0 (tuple):             Initial conditions of the limit cycle
                T (float):              Time period of the limit cycle
    """
    sol = fsolve(lambda U: shoot(ODE, U, *args), U0)
    u0 = sol[:-1]
    T = sol[-1]
    return u0, T