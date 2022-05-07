import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
from ODE_solver import test_inputs



def shoot(ODE):
        
        def G(U, pc, *args):
            """
            Returns the difference between the initial conditions and the solution to the ODE,
            and finds the phase condition.
                Parameters:
                        U(tuple):               Tuple containing the estimate of the starting conditions and time period
                        pc(function)            The phase condition function specified to be used
                        args:                   Any additional arguments required for the ODE
            """    
            # unpack U
            u0 = U[:-1]
            T = U[-1]

            # check shape
            if len(u0) > 1:
                system = True
            else:
                system = False

            sol = os.solve_ode(ODE, os.rk4_step, [0, T], u0, 0.01, system,*args)
            g1 = u0 - sol[-1]
            g2 = pc(u0, *args)   #ODE(u0, 0,*args)[pc] # phase condition
            gsol = np.append(g1, g2)

            return gsol

        return G


def limit_cycle(ODE ,U0, pc,*args):
    """
    Solves the shooting problem to find the roots to G, hence u0 and T which are
    the conditions to make a limit cycle
        Parameters:
                ODE (function):         The ODE of which the limit cycle is to be found
                U0 (tuple):             The estimated initial conditions and time period of the ODE limit cycle
                pc(function)            The phase condition function specified to be used
                args:                   Any additional arguments required for the ODE
                
        Returns:
                u0 (tuple):             Initial conditions of the limit cycle
                T (float):              Time period of the limit cycle
    """
    # test inputs
    # test ODE is a function:
    test_inputs(ODE, 'ODE', 'test_function')
    # test U0 is a tuple:
    test_inputs(U0,'U0', 'test_tuple')
    # test pc is a function
    test_inputs(pc,'pc','test_function')
    

    
    sol = fsolve(shoot(ODE),U0,args=(pc,*args))        
    return sol          


    