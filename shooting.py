import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
from ODE_solver import test_inputs



def shoot(ODE):
    """ 
    Function taking only the ODE as argument, contaning G. Allowing it to be a called as the argument
    for fsolve, with the parameters of G to be the arguments.
        Parameters:
                ODE (function):     ODE for which the shooting problem is to be solved
        """

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
        g2 = pc(u0, *args)   
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
    # test U0 is a tuple:
    test_inputs(U0,'U0', 'test_tuple')
    # test pc is a function
    test_inputs(pc,'pc','test_function')

    # test ODE is a function:
    if callable(ODE):
        # test f returns valid output
        t_test = U0[-1]
        test_output = ODE(U0[:-1], t_test, *args)
        # test valid output type
        if isinstance(test_output, (int, float ,np.int_, np.float_, list, np.ndarray)):
            # test valid output size
            if np.array(test_output).shape == np.array(U0[:-1]).shape:
               pass
            else:
                raise ValueError(f"Invalid output shape from ODE function {ODE},\n"
                                "x0 and {ODE} output should have the same shape")
        else:
            raise TypeError(f"Output from ODE function is a {type(test_output)},\n"
            "output should be an integer, float, list or array")
    else: 
        raise TypeError(f"{ODE} is not a valid input type. \n" 
                            "Please input a function")

    
    sol = fsolve(shoot(ODE),U0,args=(pc,*args))        
    return sol          


    