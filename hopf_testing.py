import numpy as np
import matplotlib.pyplot as plt
import ODE_solver as os
from scipy.optimize import fsolve
from scipy.integrate import odeint
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


def hopf_3d(u,t, beta, sigma = -1.0):

    u1 = u[0]
    u2 = u[1]
    u3 = u[2]


    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    du3dt = -u3

    return np.array((du1dt, du2dt, du3dt))



def true_hopf_3d(t, beta, theta =  0.0):

    u1 = np.sqrt(beta) * np.cos(t + theta)
    u2 = np.sqrt(beta) * np.sin(t + theta)
    u3 = np.exp(-t + beta)

    return np.array((u1, u2, u3))

t = np.linspace(0,10,101)
sol = os.solve_ode(hopf, os.rk4_step, t, (1.4,0), 0.001, 2)
u1 = sol[:,0]
u2= sol[:,1]
plt.plot(t,u1)
plt.plot(t,u2)
plt.show()


   


class Test_shooting(unittest.TestCase):

    def setUp(self):
        # 2d testing
        # find limit cycle using shooting function
        self.u0, self.T = shooting.limit_cycle(hopf, (-1,0), 5, 1.0)
        
        # use limit cycle to get solution to hopf over the cycle
        t = np.linspace(0,self.T,51)
        sol = os.solve_ode(hopf, os.rk4_step, t, self.u0, 0.001, 1.0)
        self.u1 = sol[:,0]
        self.u2 = sol[:,1]
       
        # get solution to the true hopf over the same t
        self.u1_true, self.u2_true = true_hopf(t, beta=1, theta=np.pi)

#------------------------------------------------------------------------------------------#
        
        # 3d testing
        # find limit cycle using shooting function
        self.u0_3d, self.T_3d = shooting.limit_cycle(hopf_3d, (-1,0,1), 5, 1.0)

        # use limit cycle to get solution to hopf over the cycle
        t_3d = np.linspace(0,self.T_3d,51)
        sol_3d = os.solve_ode(hopf_3d, os.rk4_step, t_3d, self.u0_3d, 0.001, 1.0)
        self.u1_3d = sol_3d[:,0]
        self.u2_3d = sol_3d[:,1]
        self.u3_3d = sol_3d[:,2]

        # get solution to the true hopf over the same t
        self.u1_true_3d, self.u2_true_3d, self.u3_true_3d = true_hopf_3d(t_3d, beta=1, theta=np.pi)
        

        
        

    def test_u0(self):
        # test same initial conditions are found
        self.assertAlmostEqual(self.u0[0], self.u1_true[0])
        self.assertAlmostEqual(self.u0[1], self.u2_true[0])
        # 3d test
        self.assertAlmostEqual(self.u0_3d[0], self.u1_true_3d[0])
        self.assertAlmostEqual(self.u0_3d[1], self.u2_true_3d[0])
        self.assertAlmostEqual(self.u0_3d[2], self.u2_true_3d[0])


    def test_period(self):
        # test the period found is 2*pi
        self.assertAlmostEqual(self.T, np.pi*2)
        # 3d test
        self.assertAlmostEqual(self.T_3d, np.pi*2)

    def test_cycle(self):
        # test that the endpoints of the cycle = the initial conditions
        self.assertAlmostEqual(self.u1[0], self.u1[-1])
        self.assertAlmostEqual(self.u2[0], self.u2[-1])
        # 3d test 
        self.assertAlmostEqual(self.u1_3d[0], self.u1_3d[-1])
        self.assertAlmostEqual(self.u2_3d[0], self.u2_3d[-1])
        self.assertAlmostEqual(self.u3_3d[0], self.u3_3d[-1])

    def test_limit_cycle(self):
        # test that all calculated points fit the true solution
        self.assertAlmostEqual(self.u1.all(), self.u1_true.all())
        self.assertAlmostEqual(self.u2.all(), self.u2_true.all())
        # 3d test
        self.assertAlmostEqual(self.u1_3d.all(), self.u1_true_3d.all())
        self.assertAlmostEqual(self.u2_3d.all(), self.u2_true_3d.all())
        self.assertAlmostEqual(self.u3_3d.all(), self.u3_true_3d.all())


if __name__ == '__main__':
    unittest.main()
