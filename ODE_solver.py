import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
""" initial conditions """
X0 = [1]
t0 = 0
h = 0.00011
deltat_max = 1
t = np.linspace(0,2,3)

'initial vector conditions'
#x0 = 0
#a0 = 1
#X0 = [x0, a0]




def f(t, x):
    
    return x



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

"""
def f(t, X):
    #print(X)
    #print(type(X))
    x = X[0]
    a = X[1]
    dxdt = a
    dadt = -x
    dXdt = [dxdt, dadt]
    
    return dXdt
"""






def euler_step(f, t, X0, h):
    """ make a single Euler step """
    x = X0 + h*np.array(f(t, X0))
    t = t + h
    return x, t



def solve_to(f, method, t0, t1, h, x0, deltat_max):
    """ loop through the euler function between t1 and t2"""
    t = t0
    x = f(t0, x0)
    f_array = []
    f_array.append(x)
    space = t1-t
    if h > deltat_max:
        return print(' step value too high')
    else:
        remainder = space%h
        repeats = (space - remainder)/h

        for i in range(int(repeats)):
            x, t = method(f, t, x, h)
            f_array.append(x)

        if remainder != 0:
            x, t = method(f, t, x, remainder)
            f_array.append(x)

    return x



def rk4_step(f, t, x, h):
    """ make a single rk4 step """
    k1 = np.array(f(t, x))
    k2 = np.array(f((t+h/2), (x+h*k1/2)))
    k3 = np.array(f((t+h/2), (x+h*k2/2)))
    k4 = np.array(f((t+h), (x+h*k3)))
    k = x + h*(k1+2*k2+2*k3+k4)/6
    t = t + h
    return k, t


    

def solve_ode(f, method ,t , x0, h, deltat_max):
    
    t0 = t[0]
    sol_array = np.zeros((len(t), len(x0)))

    sol_array[0]= x0

    for i in range(1, len(t)):
        
        t0 = t[i-1]
        t1 = t[i]
        sol_array[i] = solve_to(f, method, t0, t1, h, sol_array[i-1], deltat_max)
                    #  solve_to(f, method, t0, t1, h, x0, deltat_max)
        
    return sol_array


def plot_error():
    t0 = 0
    t1 = 1
    x0 = 1

    true_x = math.exp(1)
    h = np.linspace(0.00005, 0.1, num = 4000)
    err_euler = []
    err_rk4 = []
    
    for i in h:
        x_euler = solve_to(t0, t1, x0, i, deltat_max)
        #print(x, true_x)
        error = abs((true_x - x_euler))
        #print(error)
        err_euler.append(error)
    
        x_rk4 = solve_to_rk4(t0, t1, x0, i, deltat_max)
        #print(x, true_x)
        error = abs((true_x - x_rk4))
        #print(error)
        err_rk4.append(error)

    y_axis = err_euler
    y_axis2 = err_rk4
    x_axis = list(h)


    plt.loglog(x_axis, y_axis, marker='.', markersize=4, color='b', label='euler error')
    plt.loglog(x_axis, y_axis2, marker='.', markersize=4, color='r', label='rk4 error')
    #plt.plot(x_axis, y_axis, 'ro', linewidth=0.5)
    plt.ylabel('Error')
    plt.xlabel('Step Size')
    plt.legend()
    plt.show()

    


def plot_system(f, t, x0):
    
    X = solve_ode(f, t, x0)
    print(X)
    for i in range(len(t)-1):
        points = X[:,i]
        plt.plot(t, points)
    plt.show()

        
    
    return

    

    
    


if __name__ == '__main__':
    x = solve_ode(f, rk4_step, t, X0, h, deltat_max)
    #x = rk4_step(1, [0, 1], 0.5)

    print(x)
    #plot_error()
    #plot_system(f, rk4_step, t, X0, deltat_max)


