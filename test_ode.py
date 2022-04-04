import ODE_solver as os
import math
import numpy as np
import matplotlib.pyplot as plt


    ### Exercise 1 ###

# dx/dt = x #
def f(t, x):
    
    return x

# Initial conditions: #
x0 = [1]
t = np.linspace(1,2,2)
h=np.logspace(-6,0,100)#h = np.linspace(0.00005, 0.5, 5000)
true_x = math.exp(1)
deltat_max = 1

#  plot Euler error against rk4 error  #
def plot_error(f, t, h, x0, true_x, deltat_max):
    err_euler = np.zeros(len(h))
    err_rk4 = np.zeros(len(h))
    t0 = t[0]
    t1 = t[1]

    for i in range(0, len(h)):
        x_euler = os.solve_to(f, os.euler_step, t0, t1, h[i], x0, deltat_max)
        err_euler[i] = abs(true_x - x_euler)

        x_rk4 = os.solve_to(f, os.rk4_step, t0, t1, h[i], x0, deltat_max)
        err_rk4[i] = abs(true_x - x_rk4)

    y_axis = err_euler
    y_axis2 = err_rk4
    x_axis = list(h)

    plt.loglog(x_axis, y_axis, marker='.', markersize=4, color='b', label='euler error')
    plt.loglog(x_axis, y_axis2, marker='.', markersize=4, color='r', label='rk4 error')
    plt.ylabel('Error')
    plt.xlabel('Step Size')
    plt.legend()
    plt.show()

#plot_error(f, t, h, x0, true_x, deltat_max)




    ###  Exercise 3  ###
#  initial conditions
x0 = 0
a0 = 1
X0 = [x0, a0]
h = 0.0001
t = np.linspace(0,math.pi,500)



# d2x/dt2 = -x #
def f(t, X):
    x = X[0]
    a = X[1]
    dxdt = a
    dadt = -x
    dXdt = [dxdt, dadt]
    
    return dXdt


def plot_system(f, t, x0, h, deltat_max):
    
    X = os.solve_ode(f, os.rk4_step, t, x0, h, deltat_max)
    x = X[:,0]
    xdot = X[:,1]
    
    # x against t
    plt.plot(t, x)
    plt.show()

    # x against xdot
    plt.plot(xdot, x)
    plt.show()

plot_system(f, t, X0, h, deltat_max)
    
