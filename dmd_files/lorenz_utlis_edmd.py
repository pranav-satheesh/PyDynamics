import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


sigma = 10
beta = 8/3
rho = 28 

#Derivative function to work with RK4 loop
def lorenz(r,t):
    x = r[0]
    y = r[1]
    z = r[2]
    return np.array([sigma * (y - x), x * (rho - z) - y, (x * y) - (beta * z)])


def lorenz_evol(t0,tf,dt,r0):

    time = t0
    t = [t0] 
    h = dt
    r = r0
    r_value = [r0]
    while(t<=tf):
        #RK4 Step method
        k1 = h*lorenz(r,t)
        k2 = h*lorenz(r+k1/2,t+h/2)
        k3 = h*lorenz(r+k2/2,t+h/2)
        k4 = h*lorenz(r+k3,t+h)
        r += (k1+2*k2+2*k3+k4)/6
        r_value.append(r)
        time = time+ h
        t.append(time)
    
    return t,r


