import numpy as np
import shooting
import ODE_solver as os
from scipy.integrate import solve_ivp



# ode to solve using shooting
def hopf(t, u):

    u1, u2 = u
    beta = 1
    sigma = -1

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))



# exact solution to differential equation
def true_hopf(t):
    beta = 1
    theta = 0.0

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)

    return np.array([u1, u2])



# phase condition function for the hopf
def pc(u0, *args):
    return hopf(0, u0, *args)[0]


def hopf_3d(t, u):

    beta = 1
    sigma = -1

    u1, u2, u3 = u


    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    du3dt = -u3

    return np.array((du1dt, du2dt, du3dt))


def true_hopf_3d(t):

    beta = 1
    theta = 0

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)
    u3 = np.exp(-t)

    return np.array((u1, u2, u3))

# phase condition function for the hopf_3d
def pc3d(u0, *args):
    return hopf_3d(0, u0, *args)[0]




# test found limit cycle against exact values
def test_limit_cycle(test_f, true_f, U0, pc, tolerance, endpoints,*args):
    """
    Function testing limit cycle solutions against the true value at a given tolerance, 
    can be specified to output the closeness of the initial conditions and endpoints of the 
    limit cycle to test if a full loop is formed"""

    # find limit cycle conditions of given function to test
    soll = shooting.limit_cycle(test_f, U0, pc, *args)
    # unpack solution
    u0 = soll[:-1]
    T = soll[-1]

    # Use solve_ivp to get limit cycle values
    sol = solve_ivp(test_f,(0,T),u0)
    t_sol = sol.t
    u_sol = sol.y

    # transpose array to fit true_f dimensions
    u_sol = u_sol.T

    # get array shape to create according zero array
    rows, columns = u_sol.shape
    exact_sol = np.zeros([rows, columns])

    # get true solutions to the ode for the same times.
    for i in range(0,len(t_sol)):
        exact_sol[i] = true_f(t_sol[i])


    if endpoints == False:
        # compare true solutions to solutions obtained using the limit cycle.
        return np.allclose(u_sol, exact_sol, tolerance)
    elif endpoints == True:
        # compare initial conditions with limit cycle endpoints
        return np.allclose(exact_sol[0,:], exact_sol[-1,:], tolerance)




def test_period(test_f, U0, pc, true_T, tolerance,*args):
    """
    Function testing the limit cycle returns an accurate time period for the limit cycle"""
    # find limit cycle conditions of given function to test
    soll = shooting.limit_cycle(test_f, U0, pc, *args)
    # unpack solution
    T = soll[-1]
    return np.isclose(T, true_T, tolerance)

    




    


if __name__ == '__main__':

    print('\n')
    
    tolerance = np.logspace(-1,-21,21)  # tolerance range to test for
    
    # initial condition estimates
    U0_2d = (-1,0,6) 
    U0_3d = (-1,0,1,6)

    # test found limit cycle matches true values at same time points
    # test 2D ode
    for i in range(0,len(tolerance)):
        passed = test_limit_cycle(hopf,true_hopf,U0_2d,pc,tolerance[i],False)
        if passed == False:
            print(f"Limit cycle is found to be accurate at a tolerance level of {tolerance[i-1]} for a 2 dimensional ODE\n")
            break
        elif i == len(tolerance):
            print(f"Limit cycle is found to be accurate at a tolerance level of {tolerance[-1]} or greater, for a 2 dimensional ODE\n")

    # test 3D ode
    for i in range(0,len(tolerance)):
        passed = test_limit_cycle(hopf_3d,true_hopf_3d,U0_3d,pc3d,tolerance[i],False)
        if passed == False:
            print(f"Limit cycle is found to be accurate at a tolerance level of {tolerance[i-1]} for a 3 dimensional ODE\n")
            break
        elif i == len(tolerance):
            print(f"Limit cycle is found to be accurate at a tolerance level of {tolerance[-1]} or greater, for a 3 dimensional ODE\n")
        



    # test limit cycle is complete: test initial conditions match endpoints
    # test 2D ode
    for i in range(0,len(tolerance)):
        passed = test_limit_cycle(hopf,true_hopf,U0_2d,pc,tolerance[i],True)
        if passed == False:
            print(f"Limit cycle initial conditions found to match endpoints accurate at a tolerance level of {tolerance[i-1]} for a 2 dimensional ODE\n")
            break
        elif i == len(tolerance):
            print(f"Limit cycle initial conditions found to match endpoints accurate at a tolerance level of {tolerance[-1]} or greater, for a 2 dimensional ODE\n")


    # test 3D ode
    for i in range(0,len(tolerance)):
        passed = test_limit_cycle(hopf_3d,true_hopf_3d,U0_3d,pc3d,tolerance[i],True)
        if passed == False:
            print(f"Limit cycle initial conditions found to match endpoints accurate at a tolerance level of {tolerance[i-1]} for a 3 dimensional ODE\n")
            break
        elif i == len(tolerance):
            print(f"Limit cycle initial conditions found to match endpoints accurate at a tolerance level of {tolerance[-1]} or greater, for a 3 dimensional ODE\n")
        else:
            print('Limit cycle endpoints did not match endpoints, hence no limit cycle found')    


    # test correct Time period for the limit cycle is found
    # test 2D ode
    for i in range(0,len(tolerance)):
        test_period(hopf, U0_2d, pc, np.pi*2, tolerance)
        if passed == False:
            print(f"Limit cycle finds the time period accurate at a tolerance level of {tolerance[i-1]} for a 2 dimensional ODE\n")
            break
        elif i == len(tolerance):
            print(f"Limit cycle finds the time period accurate at a tolerance level of {tolerance[-1]} or greater for a 2 dimensional ODE\n")


    # test 3D ode
    for i in range(0,len(tolerance)):
        test_period(hopf_3d, U0_3d, pc3d, np.pi*2, tolerance)
        if passed == False:
            print(f"Limit cycle finds the time period accurate at a tolerance level of {tolerance[i-1]} for a 3 dimensional ODE\n")
            break
        elif i == len(tolerance):
            print(f"Limit cycle finds the time period accurate at a tolerance level of {tolerance[-1]} or greater for a 3 dimensional ODE\n")