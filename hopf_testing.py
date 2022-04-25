import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.optimize import fsolve
import shooting
import unittest


def hopf(u, t, beta, sigma = -1.0):

    u1 = u[0]
    u2 = u[1]

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)

    return np.array((du1dt, du2dt))


def true_hopf(t, beta, theta = 0.0):

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)

    return np.array([u1, u2])

"""
orbit = shooting.limit_cycle(hopf, (-1,0), 6, 1.0)
T = orbit[-1]
print(orbit)




t = np.linspace(0,20,401) 
# plot for visualisation
sol, sol2 = true_hopf(t, beta=1, theta=np.pi)
plt.plot(t, sol)
plt.plot(t, sol2)
plt.show()
"""

class Testshooting(unittest.TestCase):

    def setUp(self):
        # find limit cycle using shooting function
        self.u0, self.T = shooting.limit_cycle(hopf, (-1,0), 5, 1.0)

        # use limit cycle to get solution to hopf over the cycle
        t = np.linspace(0,self.T,51)
        sol = os.solve_ode(hopf, os.rk4_step, t, self.u0, 0.001, 1.0)
        self.u1 = sol[:,0]
        self.u2 = sol[:,1]
        """
        plt.plot(t, self.u1, label = 'hopf u1')
        plt.plot(t, self.u2, label = 'hopf u2')
        plt.legend()
        plt.show()
        """
        # get solution to the true hopf over the same t
        self.u1_true, self.u2_true = true_hopf(t, beta=1, theta=np.pi)
        """
        plt.plot(t, self.u1_true, label = 'true hopf u1')
        plt.plot(t, self.u2_true, label = 'true hopf u2')
        plt.legend()
        plt.show()
        """

    def test_u0(self):
        # test same initial conditions are found
        self.assertAlmostEqual(self.u0[0], self.u1_true[0])
        self.assertAlmostEqual(self.u0[1], self.u2_true[0])

    def test_period(self):
        # test the period found is 2*pi
        self.assertAlmostEqual(self.T, np.pi*2)

    def test_cycle(self):
        # test that the endpoints of the cycle = the initial conditions
        self.assertAlmostEqual(self.u1[0], self.u1[-1])
        self.assertAlmostEqual(self.u2[0], self.u2[-1])

    def test_limit_cycle(self):
        # test that all calculated points fit the true solution
        self.assertAlmostEqual(self.u1.all(), self.u1_true.all())
        self.assertAlmostEqual(self.u2.all(), self.u2_true.all())

    
    

if __name__ == '__main__':
    unittest.main()