""" initial conditions """
x0 = 1
t0 = 0
h = 1
deltat_max = 2



def f(x, t):
    return x



def euler_step(x, t, h):
    """ make a single Euler step """
    x = x + f(x, t)
    return x



def solve_to(t0, x0, h, t1, deltat_max):
    """ loop through the euler function between t1 and t2"""
    time = t0
    x = f(x0, t0)
    f_array = []
    f_array.append(x)
    space = t2 - t1
    if h > deltat_max:
        break
    else:
        repeats = space/h

        for i in len(repeats):
            x = euler_step(x, t, h)
            time += h
            f_array.append(x)

            

    return



def solve_ode():
    return


