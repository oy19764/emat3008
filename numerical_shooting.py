import math
import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.integrate import odeint

## Exercise 1  ##

def f(X, t, a=1, b=0.20, d=0.1):

    x = X[0]
    y = X[1]

    dxdt = x*(1-x)-((a*x*y)/(d+x))
    dydt = b*y*(1-(y/x))
    
    return np.array((dxdt, dydt))



# initial conditions
x0 = [3,2]
t = np.linspace(0,400,401)
h = 0.001



sol = os.solve_ode(f, os.rk4_step, t, x0, h)
#sol = odeint(fpred, v0, t, args=(a,b,d))
#print(a)


# plot time series

x = sol[:,0]
y = sol[:,1]

plt.plot(t, x, y)
plt.show()
