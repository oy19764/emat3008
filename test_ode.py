import ODE_solver as os
import math
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

    ### Exercise 1 ###

# dx/dt = x #
def f(x, t):
    
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
t = np.linspace(0,1,1000)
# time euler method
starttime_e = timer()
os.solve_ode(f,os.euler_step,t,0,euler_intercept,system=False)
endttime_e = timer()
# time rk4 method
starttime_rk4 = timer()
os.solve_ode(f,os.rk4_step,t,0,rk4_intercept,system=False)
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
def f(X, t):
    x = X[0]
    a = X[1]
    dxdt = a
    dadt = -x
    dXdt = [dxdt, dadt]
    
    return dXdt


def plot_system(f, t, t2, x0, h):
    
    X = os.solve_ode(f, os.rk4_step, t, x0, h, system=True)
    x = X[:,0]
    xdot = X[:,1]


    X2 = os.solve_ode(f, os.rk4_step, t2, X[-1], h, system=True)
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
    
