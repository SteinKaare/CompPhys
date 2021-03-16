import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

def heun(y, t, h, f, param):
    y_p = y + h * f(t, y, param)
    return y + h/2 * (f(t, y, param) + f(t + h, y_p, param))

def euler(y, t, h, f, param):
    return y + h * f(t, y, param)

def integrator(Tmax, y0, t0, h, f, param, method = heun): #Calculates time evolution of spin j, given its initial state y0
    N = int(Tmax/h)
    Y0 = np.array(y0)
    Y = np.zeros((N+2, Y0.size))
    T = np.zeros(N+2)
    #Initial conditions
    Y[0, :] = Y0
    T[0] = t0
    for i in trange(N+1):
        h = min(h, Tmax-T[i])
        Y[i+1, :] = method(Y[i, :], T[i], h, f, param)
        T[i+1] = T[i] + h
    return Y, T

def LLG(t, S, param):
    gamma, mu, a, H = param
    return - gamma / (mu * (1 + a**2)) * (np.cross(S, H) + a * np.cross(S, np.cross(S,H)))

def H_eff(S, j, J, dz, mu, B): #Here, S is the complete vector of spins and j is the spin index
    if len(S) == 1: #Just one spin
        return 2 * dz * S[j, 2] + mu * B
    if j == 0: #If first spin
        return 1/2 * J * (S[j+1]) + 2 * dz * S[j, 2] + mu * B
    if j == len(S)-1: #If last spin
        return 1/2 * J * (S[j-1]) + 2 * dz * S[j, 2] + mu * B
    return 1/2 * J * (S[j-1] + S[j+1]) + 2 * dz * S[j, 2] + mu * B