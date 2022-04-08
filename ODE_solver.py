import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def euler_step(f, x0, t0, h, *args):
    """ make a single Euler step """
    #print('before step')
    #print(x0)
    x1 = x0 + h*np.array(f(x0, t0, *args))
    t1 = t0 + h
    #print('after step')
    #print(x1)
    return x1, t1


def rk4_step(f, x0, t0, h, *args):
    """ make a single rk4 step """
    k1 = np.array(f(x0, t0, *args))
    k2 = np.array(f((x0+h*k1/2), (t0+h/2), *args))
    k3 = np.array(f((x0+h*k2/2), (t0+h/2), *args))
    k4 = np.array(f((x0+h*k3), (t0+h), *args))
    x1 = x0 + h*(k1+2*k2+2*k3+k4)/6
    t1 = t0 + h
    return x1, t1


def solve_to(f, method, t0, t1, h, x0, deltat_max, *args):
    """ loop through the euler function between t1 and t2"""
    t = t0
    #x = f(x0, t0, *args)
    f_array = []
    #f_array.append(x)
    space = t1-t
    x=x0
    if h > deltat_max:
        return print(' step value too high')
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


def solve_ode(f, method ,t , x0, h, *args):
    deltat_max = 1
    
    t0 = t[0]
    sol_array = np.zeros((len(t), len(x0)))

    sol_array[0]= x0

    for i in range(1, len(t)):
        
        t0 = t[i-1]
        t1 = t[i]
        sol_array[i] = solve_to(f, method, t0, t1, h, sol_array[i-1], deltat_max, *args)
        
    return sol_array




#

