""" initial conditions """
x0 = 1
t0 = 0
h = 0.25
deltat_max = 1



def f(x, t):
    return x



def euler_step(x, t, h):
    """ make a single Euler step """
    x = x + h*f(x, t)
    return x



def solve_to(t0, x0, h, deltat_max):
    """ loop through the euler function between t1 and t2"""
    t = t0
    x = f(x0, t0)
    f_array = []
    f_array.append(x)
    space = 1
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

            

    return x, f_array



def solve_ode():
    return

if __name__ == '__main__':

    x, a = solve_to(t0, x0, h, deltat_max)

    print(x)
    print(a)


