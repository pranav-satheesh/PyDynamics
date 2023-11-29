import numpy as np
from scipy.integrate import solve_ivp
from derivative import dxdt
import sindy



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


def lorenz_err(Xi):
    Xi_actual = np.zeros(shape=(10,3))
    Xi_actual[1,0] = -10
    Xi_actual[2,0] = 10
    Xi_actual[1,1] = 28
    Xi_actual[2,1] = -1
    Xi_actual[6,1] = -1
    Xi_actual[3,2] = -8/3
    Xi_actual[5,2] = 1

    return np.sum(np.abs(Xi_actual - Xi)**2)

def lorenz_system_gen(u0,t0,tf,dt,porder,lamda,params):

    iter = 100
    t,U,Uprime = generate_data(u0,t0,tf,dt,params)
    Xi = sindy.sparse_regress_find(U,Uprime,porder,lamda,iter)
    error = lorenz_err(Xi)
    return (t,U,Xi,error)


def lorenz_system_with_noise_gen(eta,u0,t0,tf,dt,porder,lamda,params):

    iter = 100
    t,U,Uprime = generate_noisy_data(eta,u0,t0,tf,dt,params)
    Xi = sindy.sparse_regress_find(U,Uprime,porder,lamda,iter)
    error = lorenz_err(Xi)
    return (t,U,Xi,error)


#def lorenz_system_with_noise

def sindy_lorenz_model(_,uvec,xi,poly_order):

    theta_u = sindy.create_library(uvec.reshape((1,3)),poly_order)
    (m,n) = np.shape(theta_u)
    dudt = theta_u @ xi.reshape(n,3)

    return dudt
    

def sindy_lorenz_trajectory(u0,xi,poly_order,t0,tmax,dt):

    n = int(tmax/dt)
    t = np.linspace(start=t0,stop=tmax,num=n)
    trajec = solve_ivp(fun=sindy_lorenz_model,
                       t_span=(t0,tmax),
                       y0=u0,
                       t_eval=t,
                       args=(xi,poly_order))
    
    u = trajec.y.T

    return t,u

