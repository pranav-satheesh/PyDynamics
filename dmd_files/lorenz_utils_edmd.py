import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from derivative import dxdt

sigma = 10
beta = 8/3
rho = 28 

def lorenz_de(_,u,sigma,rho,beta):

    x = u[0]
    y = u[1]
    z = u[2]

    dx_dt = sigma * (y-x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return np.hstack((dx_dt,dy_dt,dz_dt))

def gen_u(t,u0,SIGMA,RHO,BETA):

    #u0 = np.array([0,8,27])
    result = solve_ivp(fun=lorenz_de,t_span=(t[0],t[-1]),y0=u0,t_eval=t,args=(SIGMA,RHO,BETA))
    u = result.y.T

    return u

def finite_diff(u,t):

    uprime = dxdt(u.T,t,kind="finite_difference",k=1).T
    return uprime

def generate_data(u0,t0,tmax,dt,params):
    
    n = int(tmax/dt)
    t = np.linspace(start=t0,stop=tmax,num=n)

    u = gen_u(t,u0,params[0],params[1],params[2])
    uprime = finite_diff(u,t)

    return (t,u,uprime)

def generate_noisy_data(eta,u0,t0,tmax,dt,params):
    
    n = int(tmax/dt)
    t = np.linspace(start=t0,stop=tmax,num=n)

    u = gen_u(t,u0,params[0],params[1],params[2])
    uprime = finite_diff(u,t)

    uprime + eta * np.random.randn()
    return (t,u,uprime)

def lorenz_dt(_,u,A0,A1,A2,A3,A4,A5,A6,A7,A8):

    x = u[0]
    y = u[1]
    z = u[2]

    dx_dt = A1*y + A0*x
    dy_dt = A2*x + A3*y + A4*x*z
    dz_dt = A5*z+A6*x*y

    return np.hstack((dx_dt,dy_dt,dz_dt))

def edmd_lorenz_trajectory(u0,A,t0,tmax,dt):

    n = int(tmax/dt)
    t = np.linspace(start=t0,stop=tmax,num=n)
    print(A)
    print(A.shape)
    trajec = solve_ivp(fun=lorenz_dt,
                       t_span=(t0,tmax),
                       y0=u0,
                       t_eval=t,
                       args=A)
    
    u = trajec.y.T

    return t,u