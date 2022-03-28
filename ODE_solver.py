import math
import numpy as np
import matplotlib.pyplot as plt
""" initial conditions """
x0 = 1
t0 = 0
h = 0.003
deltat_max = 1
odearray = [1]



def f(x, t):
    return x



def euler_step(x, t, h):
    """ make a single Euler step """
    x = x + h*f(x, t)
    return x



def solve_to(t0, t1, x0, h, deltat_max):
    """ loop through the euler function between t1 and t2"""
    t = t0
    x = f(x0, t0)
    f_array = []
    f_array.append(x)
    space = t1-t
    if h > deltat_max:
        return print(' step value too high')
    else:
        remainder = space%h
        repeats = (space - remainder)/h

        for i in range(int(repeats)):
            x = euler_step(x, t, h)
            t += h
            f_array.append(x)

        if remainder != 0:
            x = euler_step(x, t, remainder)
            t += h
            f_array.append(x)

            

    return x



def solve_ode(odearray, x0):
    t = odearray[0]
    sol_array = []
    x = f(x0, 1)
    sol_array.append(x)

    for i in range(len(odearray)-1):
        #if i != odearray[-1]:
        t0 = odearray[i]
        t1 = odearray[i+1]
        x = solve_to(t0, t1, x, h, deltat_max)
        sol_array.append(x)
        
    return x, sol_array


def plot_error():
    t0 = 0
    t1 = 1
    x0 = 1

    true_x = math.exp(1)
    h = np.linspace(0.0000001, 1, num = 100000)
    err_array = []

    for i in h:
        x = solve_to(t0, t1, x0, i, deltat_max)
        #print(x, true_x)
        error = abs((true_x - x))
        #print(error)
        err_array.append(error)

    y_axis = np.log10(err_array)
    x_axis = np.log10(list(h))

    plt.plot(x_axis, y_axis, color='red', linewidth=0.5)
    plt.ylabel('Error')
    plt.xlabel('Step Size')
    plt.show()

    

        

    

    
    


if __name__ == '__main__':

    #x, a = solve_ode(odearray, x0)

    #print(x)
    #print(a)
    plot_error()


