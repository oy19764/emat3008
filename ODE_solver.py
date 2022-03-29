import math
import numpy as np
import matplotlib.pyplot as plt
""" initial conditions """
x0 = 1
t0 = 0
h = 0.25
deltat_max = 1
odearray = [0, 1, 2]



def f(t, x):
    return x



def euler_step(t, x, h):
    """ make a single Euler step """
    x = x + h*f(t, x)
    t = t + h
    return x, t



def solve_to(t0, t1, x0, h, deltat_max):
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
            x, t = euler_step(t, x, h)
            f_array.append(x)

        if remainder != 0:
            x, t = euler_step(t, x, remainder)
            f_array.append(x)

    return x

"""
def rk4_step(x, t, h):
    k1 = h * (f(x, t))
    k2 = h * (f((x+h/2), (t+k1/2)))
    k3 = h * (f((x+h/2), (t+k2/2)))
    k4 = h * (f((x+h), (t+k3)))
    k = x + (k1+2*k2+2*k3+k4)/6
    return k
"""
def rk4_step(t, x, h):
    k1 = (f(t, x))
    k2 = (f((t+h/2), (x+h*k1/2)))
    k3 = (f((t+h/2), (x+h*k2/2)))
    k4 = (f((t+h), (x+h*k3)))
    k = x + h*(k1+2*k2+2*k3+k4)/6
    t = t + h
    return k, t





def solve_to_rk4(t0, t1, x0, h, deltat_max):
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
            x, t = rk4_step(t, x, h)
            f_array.append(x)

        if remainder != 0:
            x, t = rk4_step(t, x, remainder)
            f_array.append(x)

    return x

    


def solve_ode(odearray, x0):

    method = input("Would you like to use the Euler of Rk4 method? type in 'E' for Euler or 'R' for RK4 " )
    euler = bool
    if method == 'E':
        euler = True
    elif method == 'R':
        euler = False
    
    t = odearray[0]
    sol_array = []

    x = f(x0, 1)
    sol_array.append(x)

    for i in range(len(odearray)-1):
        #if i != odearray[-1]:
        t0 = odearray[i]
        t1 = odearray[i+1]
        if euler == True:
            x = solve_to(t0, t1, x, h, deltat_max)
            #print(t1)
        if euler == False:
            x = solve_to_rk4(t0, t1, x, h, deltat_max)
        sol_array.append(x)
        
    return x, sol_array


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

    

        

    

    
    


if __name__ == '__main__':

    #x, a = solve_ode(odearray, x0)
    #x = rk4_step(1, 0, 0.5)

    #print(x)
    #print(a)
    plot_error()


