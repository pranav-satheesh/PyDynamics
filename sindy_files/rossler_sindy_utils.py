import numpy as np
from scipy.integrate import solve_ivp
from derivative import dxdt
import sindy

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

def rossler_system_gen(u0,t0,tf,dt,porder,lamda,params):

    iter = 100
    t,U,Uprime = generate_data(u0,t0,tf,dt,params)
    Xi = sindy.sparse_regress_find(U,Uprime,porder,lamda,iter)
    return (t,U,Xi)

def lorenz_system_with_noise_gen(eta,u0,t0,tf,dt,porder,lamda,params):

    iter = 100
    t,U,Uprime = generate_noisy_data(eta,u0,t0,tf,dt,params)
    Xi = sindy.sparse_regress_find(U,Uprime,porder,lamda,iter)
    return (t,U,Xi)

def sindy_rossler_model(_,uvec,xi,poly_order):

    theta_u = sindy.create_library(uvec.reshape((1,3)),poly_order)
    (m,n) = np.shape(theta_u)
    dudt = theta_u @ xi.reshape(n,3)

    return dudt

def sindy_rossler_trajectory(u0,xi,poly_order,t0,tmax,dt):

    n = int(tmax/dt)
    t = np.linspace(start=t0,stop=tmax,num=n)
    trajec = solve_ivp(fun=sindy_rossler_model,
                       t_span=(t0,tmax),
                       y0=u0,
                       t_eval=t,
                       args=(xi,poly_order))
    
    u = trajec.y.T

    return t,u