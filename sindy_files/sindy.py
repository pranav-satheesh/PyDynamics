import numpy as np
import matplotlib.pyplot as plt


def create_library(U,poly_order=2):

    (m,n) = U.shape
    theta = np.ones((m,1))

    #polynomial_order_1
    theta = np.hstack((theta,U))

    #polynomial_order_2
    if poly_order >= 2:
        for i in range(n):
            for j in range(i,n):
                theta = np.hstack((theta,U[:,i:i+1]*U[:,j:j+1]))

    #polynomial_order_3
    if poly_order>=3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    theta = np.hstack((theta,U[:,i:i+1]*U[:,j:j+1]*U[:,k:k+1]))


    #polynomial_order_4
    if poly_order>=4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for l in range(k,n):
                        theta = np.hstack((theta,U[:,i:i+1]*U[:,j:j+1]*U[:,k:k+1]*U[:,l:l+1]))


    return theta



def regress(theta,uprime,threshold,max_iter):

    xi = np.linalg.lstsq(theta,uprime,rcond=None)[0]
    n = xi.shape[1]

    for _ in range(max_iter):
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0
        
        for j in range(n):
            big_indices = np.logical_not(small_indices[:,j])
            xi[big_indices,j] = np.linalg.lstsq(theta[:,big_indices],uprime[:,j],rcond=None)[0]

    return xi


def sparse_regress_find(U,Uprime,porder,threshold,N):

    Theta = create_library(U,porder)
    Xi = regress(Theta,Uprime,threshold,N)

    return Xi


