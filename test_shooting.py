import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.optimize import fsolve

## Exercise 1  ##

def f(X, t, a, b, d):

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
#plt.plot(x, y)
#plt.ylabel('y')
#plt.xlabel('x')
#plt.show()

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
    print(orbit_limit)
    print(period_list)

    start = intersect[-2]
   
    t_orb = np.linspace(period_list[-2],period_list[-1],200)
    orbitsol = os.solve_ode(f, os.rk4_step, t_orb, start, h, *args)

    orbit_x = orbitsol[:,0]
    orbit_y = orbitsol[:,1]
    
    #plt.plot(x, y)
    #plt.plot(orbit_x, orbit_y, label='Orbit isolated')
    #plt.ylabel('y')
    #plt.xlabel('x')
    #plt.legend()
    #plt.show()    

    return period, orbit_x, orbit_y




# isolated period orbit
period, orbit_x, orbit_y = find_orbit(f, t, x, y, h, a, b, d)
#period
print('The period of the orbit is:  {}'.format(period))



## Exercise 2  ##

# An appropriate phase condition would be to use either derivative of x or y at time 0
# dxdt(0) or dydt(0) can be used.

## Exercise 3  ##


T = 22 # period estimate
U0 = [0.4, 0.4] # start conditions estimate
#print(phase_condition(U0, a, b, d))
#print(G(U0, 0, T, a, b, d))
#a = fsolve(G, [0.3, 0.32], args=(a,b,d))
#print(a)



def orbit(u0, T, *args):

    
    def shoot(U, *args):

        def G(u0, T, *args):

            sol = os.solve_ode(f, os.rk4_step, [0, T], u0, 0.001, *args)
            g1 = u0 - sol[-1]
            g2 = f(u0, t, *args)[1] # phase condition
            gsol = np.append(g1, g2)
            return gsol


        u0 = U[:-1]
        T = U[-1]
        shot = G(u0, T, *args)

        return shot


    sol = fsolve(lambda U: shoot(U, *args), np.concatenate((u0, [T])))
    u0 = sol[:-1]
    T = sol[-1]
    return u0, T

"""
u0, T = orbit(U0, T, a, b, d)

sol_orbit = os.solve_ode(f,  os.rk4_step, np.linspace(0, T, 200), u0, 0.001, a, b, d)

plt.plot(sol_orbit[:,0], sol_orbit[:,1], label='Orbit isolated')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()    
"""
import shooting

print(shooting.limit_cycle(f, U0, T, a, b, d))

