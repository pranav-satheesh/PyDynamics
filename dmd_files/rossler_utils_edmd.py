import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def rossler(r,t):
    x = r[0]
    y = r[1]
    z = r[2]
    return np.array([- y - z, x + a * y, b + z * (x - c)])