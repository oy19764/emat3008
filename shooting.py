import numpy as np
import ODE_solver as os
from scipy.optimize import fsolve
from parameter_tests import test_inputs, test_ode



def shoot(ODE):
    """ 
    Function taking only the ODE as argument, contaning G. Allowing it to be a called as the argument
    for fsolve, with the parameters of G to be the arguments.
        Parameters:
                ODE (function):     ODE for which the shooting problem is to be solved
        Output:
                G:                  Array containing estimates for starting condions and the time period
        """

    def G(U, pc, *args):
        """
        Returns the difference between the initial conditions and the solution to the ODE,
        and finds the phase condition.
            Parameters:
                    U(tuple):               Tuple containing the estimate of the starting conditions and time period
                    pc(function)            The phase condition function specified to be used
                    args:                   Any additional arguments required for the ODE
            Output:
                    gsol:                   Array containing estimates for starting condions and the time period
        """    
        # unpack U
        u0 = U[:-1]
        T = U[-1]
        

        # check shape 
        if len(u0) > 1:
            system = True
        else:
            system = False

        # solve ode for given estimates
        sol = os.solve_ode(ODE, [0, T], u0, 0.01, system, os.rk4_step,*args)
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
    # test ODE is an ODE:
    test_ode(ODE, U0[:-1], *args)


    sol = fsolve(shoot(ODE),U0,args=(pc,*args))        
    return sol          


    

if __name__ == '__main__':

    import matplotlib.pyplot as plt


        ## Exercise 1  ##

    def f(t, X, a, b, d):

        x = X[0]
        y = X[1]
        #x, y = X

        dxdt = x*(1-x)-((a*x*y)/(d+x))
        dydt = b*y*(1-(y/x))
        
        return np.array((dxdt, dydt))


    #solving using b = 0.36 > 0.26
    a = 1
    b = 0.36
    d = 0.1


    # initial conditions
    x0 = [0.35, 0.35]
    t = np.linspace(0,100,3001)
    h = 0.001
    sol = os.solve_ode(f, t, x0, h, True, os.rk4_step, a, b, d)
    x = sol[:,0]
    y = sol[:,1]


    # plot time series
    plt.subplot(2,2,1)
    plt.plot(t, x)
    plt.plot(t, y)
    plt.ylabel('locta volterra values')
    plt.xlabel('t')
    plt.title('locta volterra time series for b > 0.26')

    # plot y against x
    plt.subplot(2,2,2)
    plt.plot(x, y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('locta volterra phase portrait for b > 0.26')


    # Solving using b = 0.16 < 0.26
    b = 0.16


    sol = os.solve_ode(f, t, x0, h, True, os.rk4_step, a, b, d)
    x = sol[:,0]
    y = sol[:,1]

    # plot time series
    plt.subplot(2,2,3)
    plt.plot(t, x)
    plt.plot(t, y)
    plt.ylabel('locta volterra values')
    plt.xlabel('t')

    plt.title('locta volterra time series for b < 0.26')
    # plot y against x
    plt.subplot(2,2,4)
    plt.plot(x, y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('locta volterra phase portrait for b < 0.26')
    plt.show()


    """
        The figures show different outcomes for the predator-prey equations depending on the value of b.
    When b > 0.26, the fluctuation in population seems to flatten out. This is reinforced by the 
    spiral seen in y against x, as both populations converge towards a point.
    When b < 0.26, the fluctuation in the populations of x and y continues to fluctuate, the plot
    of y against x shows the development of a circle, indication the formation of period orbits.
    """



    ## Isolate a periodic orbit ##

    def find_orbit(f, t, x, y, h, *args):
        period_list = []
        intersect = []
        for i in range(len(t)):
            if y[i] >= 0.37:
                if abs(x[i] - y[i]) <= 0.001:
                    period_list.append(t[i])
                    intersect.append(np.array([x[i],y[i]]))

        #orbit_limit = [period_list[-2], period_list[-1]]
        orbit_limit = np.array([period_list[-2], period_list[-1]])
        period = period_list[-1] - period_list[-2]

        start = intersect[-2]
    
        t_orb = np.linspace(period_list[-2],period_list[-1],200)
        orbitsol = os.solve_ode(f, t_orb, start, h, True, os.rk4_step,*args)

        orbit_x = orbitsol[:,0]
        orbit_y = orbitsol[:,1]
        
        plt.plot(x, y)
        plt.plot(orbit_x, orbit_y, label='Orbit isolated')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.legend()
        plt.show()    

        return period, orbit_x, orbit_y, orbitsol



    # isolated period orbit
    period, orbit_x, orbit_y, manual_orbitsol = find_orbit(f, t, x, y, h, a, b, d)

    #period
    print('The period of the orbit is:  {}'.format(period))


    ## Exercise 2  ##



    # An appropriate phase condition would be to use either derivative of x or y at time 0
    # dxdt(0) or dydt(0) can be used.

    def phase_condition(u0, *args): 
        return f(0, u0,*args)[1] # phase condition



    ## Exercise 3  ##

    T = 22 # period estimate
    u0 = [0.4, 0.4] # start conditions estimate


    # limit cycle functon taken from Exercise 2
    U0 = (0.4, 0.4, 22)
    sol = limit_cycle(f, U0, phase_condition, a, b, d)
    u0 = sol[:-1]
    T = sol[-1]


    # test orbit against manually found orbit

    t_orb = np.linspace(0,T,200)
    orbitsol = os.solve_ode(f, t_orb, u0, h, True, os.rk4_step, a, b, d)


    test = np.isclose(orbitsol, manual_orbitsol, atol=1e-03)
    if np.all(test) == True:
        print('successful estimation of limit cycle')
    else:
        print('unsuccesful estimation')