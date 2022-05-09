import numpy as np
from parameter_tests import test_inputs, test_ode


def euler_step(f, t0, x0, h, *args):
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
    x1 = x0 + h*np.array(f(t0, x0, *args))
    t1 = t0 + h
    return t1, x1


def rk4_step(f, t0, x0, h, *args):
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
    k1 = np.array(f(t0, x0, *args))
    k2 = np.array(f((t0+h/2), (x0+h*k1/2), *args))
    k3 = np.array(f((t0+h/2), (x0+h*k2/2), *args))
    k4 = np.array(f((t0+h), (x0+h*k3), *args))
    x1 = x0 + h*(k1+2*k2+2*k3+k4)/6
    t1 = t0 + h
    return t1, x1


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
    # test f is an ODE
    test_ode(f, x0, *args)


    t = t0
    space = t1-t
    x=x0
    if h > deltat_max:
        raise ValueError(f"h value: {h}, is greater than maximimum accepted step size: {deltat_max}")
    else:
        remainder = space%h
        repeats = (space - remainder)/h

        for i in range(int(repeats)):
            t, x = method(f, t, x, h, *args)

        if remainder != 0:
            t, x = method(f, t, x, remainder, *args)
            
    return x




def solve_ode(f, t, x0, h, system=False, method=rk4_step,*args):
    """ 
    Solves the ode over the time period t.
        Parameters:
            f(function):        ODE function to solve
            t(ndarray):         t values to solve the ODE for
            x0(tuple):          Initial known value of x
            h(float):           Step size
            system(boolean):    Boolean indicating wether or not f is a system of equations
            method(function):   Method to solve the ode with
            *args:              Any additional arguments required for the ODE
        Returns:
            sol_array(ndarray): Solutions to the ODE for each time value in t
    """
    
    # test parameters:
    # test method is a function
    test_inputs(method, 'method', 'test_function')
    # test t is an array
    test_inputs(t, 't', 'test_array')
    # test h is float
    test_inputs(h, 'h', 'test_int_or_float')
    # test system is bool
    test_inputs(system, 'system', 'test_bool')
    # test f is an ODE
    test_ode(f, x0, *args)
    
    
    
    # define array of solutions
    if system == True:
        # test x0 is an array
        test_inputs(x0, 'x0', 'test_array')
        sol_array = np.zeros((len(t), len(x0)))
    elif system == False:
        # test x0 is an integer or float
        test_inputs(x0, 'x0', 'test_int_or_float')
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





if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer
    import math

        ### Exercise 1 ###

    # dx/dt = x #
    def f(t, x):
        
        return x

    # Initial conditions: #
    x0 = 1
    t = np.linspace(1,2,2)
    h = np.linspace(0.00005, 0.5, 5000)
    true_x = math.exp(1)
    deltat_max = 1

    #  plot Euler error against rk4 error  #
    def plot_error(f, t, h, x0, true_x, deltat_max):
        err_euler = np.zeros(len(h))
        err_rk4 = np.zeros(len(h))
        t0 = t[0]
        t1 = t[1]

        for i in range(0, len(h)):
            x_euler = solve_to(f, euler_step, t0, t1, h[i], x0, deltat_max)
            err_euler[i] = abs(true_x - x_euler)

            x_rk4 = solve_to(f, rk4_step, t0, t1, h[i], x0, deltat_max)
            err_rk4[i] = abs(true_x - x_rk4)

        y_axis = err_euler
        y_axis2 = err_rk4
        x_axis = list(h)

        plt.loglog(x_axis, y_axis, marker='.', markersize=4, color='b', label='euler error')
        plt.loglog(x_axis, y_axis2, marker='.', markersize=4, color='r', label='rk4 error')
        plt.ylabel('Error')
        plt.xlabel('Step Size')
        plt.title('Error plot for the rk4 and euler method for increasing step size')
        plt.legend()
        #plt.show()

        # find step sizes for same error
        err_euler = np.round(err_euler,4)
        err_rk4 = np.round(err_rk4,4)
        intersect, euler_i, rk4_i = np.intersect1d(err_euler, err_rk4, return_indices=True)
        euler_intercept = np.round(h[euler_i],5)
        rk4_intercept = np.round(h[rk4_i],5)
        #plot step sizes for same error on the same plot
        plt.plot(euler_intercept[1], intersect[1], marker='o', color='y', label=f' Euler step size = {euler_intercept[1]}')
        plt.plot(rk4_intercept[1], intersect[1], marker='o', color='g', label=f' rk4 step size = {rk4_intercept[1]}')
        plt.axhline(y=intersect[1], color='k', ls='-', label=f'Error = {intersect[1]}')
        plt.legend()
        plt.show()

        return rk4_intercept[1], euler_intercept[1]
        
    rk4_intercept, euler_intercept = plot_error(f, t, h, x0, true_x, deltat_max)

    # time and compare rk4 and euler method for the same for the same step size
    # intitial conditions
    t = np.linspace(0,10,1000)
    # time euler method
    starttime_e = timer()
    solve_ode(f,t,0,euler_intercept,system=False,method=euler_step)
    endttime_e = timer()
    # time rk4 method
    starttime_rk4 = timer()
    solve_ode(f,t,0,rk4_intercept,system=False,method=rk4_step)
    endttime_rk4 = timer()
    euler_time = endttime_e - starttime_e
    rk4_time = endttime_rk4 - starttime_rk4
    print(euler_time, 'Time taken to solve ode using euler method.')
    print(rk4_time, 'Time taken to solve ode using rk4 method.')
    t_diff = euler_time - rk4_time
    print(f'The rk4 method was {t_diff} seconds faster than the euler method.')

        ###  Exercise 3  ###
    #  initial conditions
    x0 = 0
    a0 = 1
    X0 = [x0, a0]
    h = 0.7
    t = np.linspace(0,250,251)
    t2 = np.linspace(250,500,251)

    """
    x'' = -x
    equivalent to:
    x' = a
    a' = -x
    treating as a vector:
    (x, a)' = (a, -x)
    make function that can compute the rhs of the vector:
    X' = f(X, t)
    """

    # d2x/dt2 = -x #
    def f(t, X):
        x = X[0]
        a = X[1]
        dxdt = a
        dadt = -x
        dXdt = [dxdt, dadt]
        
        return dXdt


    def plot_system(f, t, t2, x0, h):
        print(x0)
        X = solve_ode(f, t, x0, h, system=True,method=rk4_step)
        x = X[:,0]
        xdot = X[:,1]
        x0 = X[-1]
        print(x0)
        X2 = solve_ode(f, t2, x0, h, system=True,method=rk4_step)
        x2 = X2[:,0]
        xdot2 = X2[:,1]

        # x against t
        plt.subplot(1,2,1)
        plt.plot(t, x, label='x for t values 0-250')
        plt.plot(t2, x2, label = 'x for t values 250-500')
        plt.ylabel('x')
        plt.xlabel('t')
        plt.title("Numerical solution x against t, for large step size")
        plt.legend()

        # x against xdot
        plt.subplot(1,2,2)
        plt.plot(xdot, x, label='x for t values 0-250')
        plt.plot(xdot2,x2,label='x for t values 250-500')
        plt.ylabel('x dot')
        plt.xlabel('x')
        plt.legend()
        plt.title("Numerical solution x' against x, for large step size")
        plt.show()

    plot_system(f, t, t2, X0, h)

    # It is shown in these plots that with large time steps and over a large range of t,
    # that the system becomes increasingly unstable
