import numpy as np
from scipy.integrate import odeint


def euler_step(f, x0, t0, h, *args):
    """ 
    Makes a single Euler step of step size h for the given function.
        Parameters:
            f(function):        ODE for which we are performing the Euler Step
            x0(tuple):          x value at which we start the step
            t0(float):          t value at which we start the step
            h(float):           Step size
            *args:              Any additional arguments required for the ODE
        Returns:
            x1:                 x values after performing the step
            t1:                 t value after performing the step
    """
    x1 = x0 + h*np.array(f(x0, t0, *args))
    t1 = t0 + h
    return x1, t1


def rk4_step(f, x0, t0, h, *args):
    """ 
    Makes a single RK4 step of step size h for the given function.
        Parameters:
            f(function):        ODE for which we are performing the RK4 step
            x0(tuple):          x value at which we start the step
            t0(float):          t value at which we start the step
            h(float):           Step size
            *args:              Any additional arguments required for the ODE
        Returns:
            x1:                 x values after performing the step
            t1:                 t value after performing the step
    """
    k1 = np.array(f(x0, t0, *args))
    k2 = np.array(f((x0+h*k1/2), (t0+h/2), *args))
    k3 = np.array(f((x0+h*k2/2), (t0+h/2), *args))
    k4 = np.array(f((x0+h*k3), (t0+h), *args))
    x1 = x0 + h*(k1+2*k2+2*k3+k4)/6
    t1 = t0 + h
    return x1, t1


def solve_to(f, method, t0, t1, h, x0, deltat_max, *args):
    """ 
    Loop through the specified solver method between t0 and t1 
    in steps no bigger than deltat_max.
        Parameters:
            f(function):        ODE function to solve
            method(function):   Method to solve the ode with
            t0(float):          initial t value at which we start solving
            t1(float):          Time value to solve to
            h(float):           Step size
            x0(tuple):          Initial x value at which we start solving
            deltat_max(float):  Maximum step size permitted
            *args:              Any additional arguments required for the ODE
        Returns:
            x:                  x values at time t1
    """
    # test parameters:
    # test f is a function
    test_inputs(f, 'f', 'test_function')
    # test method is a function
    test_inputs(method, 'method', 'test_function')
    # test t0 is an integer or float
    test_inputs(t0, 't0', 'test_int_or_float')
    # test t1 is an integer or float
    test_inputs(t1, 't1', 'test_int_or_float')
    # test h is an integer or float
    test_inputs(h, 'h', 'test_int_or_float')
    # test deltat_max is an integer or float
    test_inputs(deltat_max, 'deltat_max', 'test_int_or_float')



    t = t0
    f_array = []
    space = t1-t
    x=x0
    if h > deltat_max:
        raise ValueError(f"h value: {h}, is greater than maximimum accepted step size: {deltat_max}")
    else:
        remainder = space%h
        repeats = (space - remainder)/h

        for i in range(int(repeats)):
            x, t = method(f, x, t, h, *args)
            f_array.append(x)

        if remainder != 0:
            x, t = method(f, x, t, remainder, *args)
            f_array.append(x)

    return x


def test_inputs(input, input_name, test_name, ):
    """
    Testing function to check function inputs are valid.
        Parameters:
            input:          input being tested
            input_name:     name of input being tested
            test_name:      test to check input with
    """
    # define tests
    def test_int_or_float(input, input_name):
        if not isinstance(input, (int, float, np.int_, np.float_)):
            raise TypeError(f"{input_name}: {input} is not a valid type. \n" 
                            "Please input an integer or a float")
    

    def test_function(input, input_name):
        if not callable(input):
            raise TypeError(f"{input_name} is not a valid input. \n" 
                            "Please input a function")

    
    def test_array(input, input_name):
        if not isinstance(input, np.ndarray):
            raise TypeError(f"{input_name} is not a valid type. \n" 
                            "Please input a numpy array")


    def test_tuple(input, input_name):
        if not isinstance(input, tuple):
            raise TypeError(f"{input_name}: {input} is not a valid type. \n" 
                            "Please input a tuple")

    def test_bool(input, input_name):
        if not isinstance(input, bool):
            raise TypeError(f"{input_name}: {input} is not a valid type. \n" 
                            "Please input a boolean ('True' or 'False')")

   
    # call test to perform
    if test_name == 'test_int_or_float':
        test_int_or_float(input, input_name)

    if test_name == 'test_function':
        test_function(input, input_name)

    if test_name == 'test_array':
        test_array(input, input_name)

    if test_name == 'test_tuple':
        test_tuple(input, input_name)

    if test_name == 'test_bool':
        test_bool(input, input_name)




def solve_ode(f, method ,t , x0, h, system=False,*args):
    """ 
    Solves the ode over the time period t.
        Parameters:
            f(function):        ODE function to solve
            method(function):   Method to solve the ode with
            t(ndarray):         t values to solve the ODE for
            x0(tuple):          Initial known value of x
            h(float):           Step size
            system(boolean):    Boolean indicating wether or f is a system of equations
            *args:              Any additional arguments required for the ODE
        Returns:
            sol_array(ndarray): Solutions to the ODE for each time value in t
    """
    # test parameters:
    # test f is a function
    test_inputs(f, 'f', 'test_function')
    # test method is a function
    test_inputs(method, 'method', 'test_function')
    # test t is an array
    test_inputs(t, 't', 'test_array')
    # test h is float
    test_inputs(h, 'h', 'test_int_or_float')
    # test system is bool
    test_inputs(system, 'system', 'test_bool')



    # define array of solutions
    if system == True:
        # test x0 is an array
        (x0, 'x0', 'test_array')
        sol_array = np.zeros((len(t), len(x0)))
    elif system == False:
        # test x0 is an integer or float
        (x0, 'x0', 'test_int_or_float')
        sol_array = np.zeros((len(t)))

    #define deltat_max
    deltat_max = 1
    t0 = t[0]
    sol_array[0]= x0
    
    for i in range(1, len(t)):
        
        t0 = t[i-1]
        t1 = t[i]
        sol_array[i] = solve_to(f, method, t0, t1, h, sol_array[i-1], deltat_max, *args)
                
    return sol_array





