import numpy as np #For arrays
import matplotlib.pyplot as plt #For plotting
from scipy.integrate import solve_ivp

cutoff = 1e-1

def trial_function_eval(r, rdot):
    # Degree one variables
    (m,n) = r.shape
    ThetaR = r
    ThetaRd = rdot

    # Degree two variables

    # x^2, xdot^2
    ThetaR = np.vstack((ThetaR, r[0,:]**2))
    ThetaRd = np.vstack((ThetaRd, rdot[0,:]**2))

    #y^2, ydot^2
    ThetaR = np.vstack((ThetaR, r[1,:]**2))
    ThetaRd = np.vstack((ThetaRd, rdot[1,:]**2))

    #z^2, zdot^2
    ThetaR = np.vstack((ThetaR, r[2,:]**2))
    ThetaRd = np.vstack((ThetaRd, rdot[2,:]**2))

    #x*y, xdot*ydot
    ThetaR = np.vstack((ThetaR, r[0,:]*r[1,:]))
    ThetaRd = np.vstack((ThetaRd, rdot[0,:]*rdot[1,:]))

    #x*z, xdot*zdot
    ThetaR = np.vstack((ThetaR, r[0,:]*r[2,:]))
    ThetaRd = np.vstack((ThetaRd, rdot[0,:]*rdot[2,:]))

    #y*z, ydot*zdot
    ThetaR = np.vstack((ThetaR, r[2,:]*r[1,:]))
    ThetaRd = np.vstack((ThetaRd, rdot[2,:]*rdot[1,:]))
    
    #Constant
    ThetaR = np.vstack((ThetaR,np.ones((1,n))))
    ThetaRd = np.vstack((ThetaRd,np.ones((1,n))))
    
    return ThetaR, ThetaRd


def reconstructed(A, r, t):
    B = np.copy(A)
    C = np.copy(A)
    B[B>(-cutoff)]=0
    C[C<cutoff]=0
    temp=C+B
    r1=[r[0],r[1],r[2],r[0]**2,r[1]**2,r[2]**2,r[0]*r[1],r[0]*r[2],r[1]*r[2]]
    temp2 = np.matmul(temp,r1)
    return temp2[0:3]

def findA(ThetaR, ThetaRd):
    A = np.matmul(ThetaRd,np.linalg.pinv(ThetaR))
    B = np.copy(A)
    C = np.copy(A)
    B[B>(-cutoff)]=0
    C[C<cutoff]=0
    temp=C+B
    return temp