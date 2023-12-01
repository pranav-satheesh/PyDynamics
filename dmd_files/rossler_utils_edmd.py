import numpy as np
from scipy.integrate import solve_ivp
from derivative import dxdt

def rossler_de(_,u,a,b,c):

    x = u[0]
    y = u[1]
    z = u[2]

    dx_dt = -y-z
    dy_dt = x  + a * y
    dz_dt = b + z*(x-c)

    return np.hstack((dx_dt,dy_dt,dz_dt))

def gen_u(t,u0,a,b,c):

    #u0 = np.array([0,8,27])
    result = solve_ivp(fun=rossler_de,t_span=(t[0],t[-1]),y0=u0,t_eval=t,args=(a,b,c))
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

def rossler_dt(_,u,A0,A1,A2,A3,A4,A5,A6):

    x = u[0]
    y = u[1]
    z = u[2]

    dx_dt = A0*y + A1*z
    dy_dt = A2*x + A3*y
    dz_dt = A4*z + A5*x*z +A6

    return np.hstack((dx_dt,dy_dt,dz_dt))

def dudt(_,u,A):
    r = np.array([u[0],u[1],u[2],u[0]**2,u[1]**2,u[2]**2,u[0]*u[1],u[0]*u[2],u[1]*u[2],1])
    return np.matmul(A,r)

def edmd_rossler_trajectory(u0,A,t0,tmax,dt):

    n = int(tmax/dt)
    t = np.linspace(start=t0,stop=tmax,num=n)
    trajec = solve_ivp(fun=dudt,
                       t_span=(t0,tmax),
                       y0=u0,
                       t_eval=t,
                       args=(A,))
    
    u = trajec.y.T

    return t,u