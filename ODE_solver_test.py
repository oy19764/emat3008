from ODE_solver import solve_ode, rk4_step, euler_step
import numpy as np


# test solution accuracy for single ode's and system of ode's

def ode(t, x):
    return x


def true_ode(t):
    return np.exp(t)


def ode_system(t, X):
    beta=1
    x, y = X
    dxdt = beta*x - y - x*(x**2 + y**2)
    dydt = x + beta*y - y*(x**2 + y**2)
    return np.array((dxdt,dydt))


def true_ode_system(t):
    beta=1
    x = np.sqrt(beta)*np.cos(t+np.pi)
    y = np.sqrt(beta)*np.sin(t+np.pi)
    return np.array((x,y))



def output_test(test_f, true_f, h, method, tolerance, system):
    """
    Function testing ode solution found using the ODE solver against its true value at
    a given tolerance level
        Parameters:
                test_f(function):       function being tested
                true_f(function)        exact value of the function
                h(float):               step size to solve at
                method(function):       solving method to be tested
                system(boolean):        boolean indicating if we are testing a system or not
                tolerance(float):       tolerance level to test at
        Output:
                boolean:                True if test is passed and False if test is failed
    """
    t = np.linspace(0,10,51)

    if system == True:
        n = len(true_f(0))    
        true_sol = np.zeros([51,n])
        for i in range(0,len(t)):
            true_sol[i] = true_f(t[i])
            x0 = true_sol[0,:]
            sol = solve_ode(test_f, t, x0, h, system, method)

    else:
        true_sol = np.zeros(51)
        for i in range(0,len(t)):
            true_sol[i] = true_f(t[i])
        x0 = true_sol[0]
        sol = solve_ode(test_f, t, x0, h, system, method)


    if np.allclose(sol, true_sol, tolerance):
        return True          
    else:
        return False 


if __name__ == '__main__':

    print('\n')
    h = 0.01 # step size to test for
    
    tolerance = np.logspace(-1,-12,12)  # tolerance range to test for

    # test for each type of ode and solving method:
    # euler method single ode test
    for i in range(0,len(tolerance)):
        passed = output_test(ode, true_ode, h, euler_step, tolerance[i], system=False)
        if passed == False:
            print(f"Euler step method is accurate at solving single ODE's up to a tolerance of {tolerance[i-1]} for step size {h}\n")
            break
        elif i == len(tolerance):
            print(f"Euler step method is accurate at solving single ODE's with a tolerance of {tolerance[-1]} or greater, for step size {h}\n")


    # euler method system of ode test
    for i in range(0,len(tolerance)):
        passed = output_test(ode_system, true_ode_system, h, euler_step, tolerance[i], system=True)
        if passed == False:
            print(f"Euler step method is accurate at solving system's of ODE's up to a tolerance of {tolerance[i-1]} for step size {h}\n")
            break        
        elif i == len(tolerance):
            print(f"Euler step method is accurate at solving system's of ODE's with a tolerance of {tolerance[-1]} or greater, for step size {h}\n")


    # rk4 method single ode test
    for i in range(0,len(tolerance)):
        passed = output_test(ode, true_ode, h, rk4_step, tolerance[i], system=False)
        if passed == False:
            print(f"4th order Runge Kutta method is accurate at solving single ODE's up to a tolerance of {tolerance[i-1]} for step size {h}\n")
            break    
        elif i == len(tolerance): 
            print(f"4th order Runge Kutta method is accurate at solving single ODE's with a tolerance of {tolerance[-1]} or greater, for step size {h}\n")   


    # rk4 method system of ode test
    for i in range(0,len(tolerance)):
        passed = output_test(ode, true_ode, h, rk4_step, tolerance[i], system=False)
        if passed == False:
            print(f"4th order Runge Kutta method is accurate at solving system's of ODE's up to a tolerance of {tolerance[i-1]} for step size {h}\n")
            break
        elif i == len(tolerance):
            print(f"4th order Runge Kutta method is accurate at solving system's of ODE's with a tolerance of {tolerance[-1]} or greater, for step size {h}\n") 
    

    