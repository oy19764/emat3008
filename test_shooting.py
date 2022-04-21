import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.integrate import odeint
from scipy.optimize import fsolve

## Exercise 1  ##

def f(X, t, a, b, d):

    #a = args[0]
    #b = args[1]
    #d = args[2]

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
"""
sol = os.solve_ode(f, os.rk4_step, t, x0, h, a, b, d)
x = sol[:,0]
y = sol[:,1]

# plot time series
plt.plot(t, x)
plt.plot(t, y)
plt.show()
# plot y against x
plt.plot(x, y)
plt.ylabel('y')
plt.xlabel('x')
plt.show()
"""
# Solving using b = 0.16 < 0.26
b = 0.16

sol = os.solve_ode(f, os.rk4_step, t, x0, h, a, b, d)
x = sol[:,0]
y = sol[:,1]
# plot time series
plt.plot(t, x)
plt.plot(t, y)
plt.show()
# plot y against x
plt.plot(x, y)
plt.ylabel('y')
plt.xlabel('x')
plt.show()

"""
    The figures show different outcomes for the predator-prey equations depending on the value of b.
When b > 0.26, the fluctuation in population seems to flatten out. This is reinforced by the 
spiral seen in y against x, as both populations converge towards a point.
When b < 0.26, the fluctuation in the populations of x and y continues to fluctuate, the plot
of y against x shows the development of a circle, indication the formation of period orbits.
"""

## Isolate a periodic orbit ##
def find_orbit(f, t, x, y, h, a, b, d):
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
    print(orbit_limit)

    start = intersect[-2]
   
    t_orb = np.linspace(period_list[-2],period_list[-1],200)
    orbitsol = os.solve_ode(f, os.rk4_step, t_orb, start, h, a, b, d)

    orbit_x = orbitsol[:,0]
    orbit_y = orbitsol[:,1]
    
    plt.plot(x, y)
    plt.plot(orbit_x, orbit_y, label='Orbit isolated')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()    

    return period


# isolated period orbit
period = find_orbit(f, t, x, y, h, a, b, d)
#period
print('The period of the orbit is:  {}'.format(period))



## Exercise 2  ##

# An appropriate phase condition would be to use either derivative of x or y at time 0
# dxdt(0) or dydt(0) can be used.

## Exercise 3  ##










#results: x = 0.14973664465794942  y = 0.15001583632703638 t = 51.300000000000004
# initial conditions:
u0 = [0.35, 0.35]     # using same known initial conitions
# hence boundary conditions : x(0) = x(T), y(0) = y(T)
# f(x,y,T)=[x(0)]-x(T), y(0)]-y(T)] = 0
# we need to add a phase condition:
# f(x,y,T)=[x(0)]-x(T), y(0)]-y(T), dxdt(0)-p] = 0



T = 21     # estimate of the time period
t0 = 0 #np.linspace(t0,100,1001)







"""
G = lambda U: [
    U[:-1] - os.solve_ode(f, os.rk4_step, [0, U[:-1]], U[-1], 0.001),
    f(U[-1], U[:-1])
]
a = fsolve(G, 20)
"""







"""
G = lambda U: [
    *(U[:-1] - os.solve_ode(f, os.rk4_step, [0, U[-1]], U[-1], 0.001, *args)[-1]),
    f(U[-1], U[:-1], *args)[0],
]
print(G)


def G(f, U, *args):
    print(U)

    u0 = U[:-1]
    T = U[-1]

    sol = os.solve_ode(f, os.rk4_step, [0, T], u0, 0.001, *args)

    gamma = f(u0, 0, *args)[1] #phase condition

    a = u0 - sol

    return np.array(a, gamma)


def solve_system(f, u0, T, *args):

    sol = fsolve(lambda U, f: G(f, U, *args), np.concatenate((u0, T)), f)

    return sol

sol2, gamma = G(f, u0, t0, T)

#print(gamma)
#print(sol2)

"""