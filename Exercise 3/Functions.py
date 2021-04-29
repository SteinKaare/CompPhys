import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.integrate import simps
from scipy.signal import gaussian
from scipy.stats import moment
from numba import njit
from tqdm import trange


def constructLR(K, dt, dz, kw):
    a = dt / (2 * dz**2)
    gamma = 2 * a * kw * dz * (1 - (-3/2 * K[0] + 2 * K[1] - 1/2 * K[2]) / (2 * K[0]))
    K_prime = K[2:] - K[:-2] #Slicing does not include the last element specified
    #Setting diagonals for the L matrix
    #Super-diagonal
    super_diag = - a * K[:-1]
    super_diag[0] -= a * K[0] #Ensures -2aK_0 on the first element of this diagonal
    super_diag[1:] -= a/4 * K_prime
    #Main diagonal
    main_diag = 1 + 2 * a * K
    main_diag[0] += gamma
    #Sub_diagonal
    sub_diag = - a * K[1:]
    sub_diag[-1] -= a * K[-1] #Ensures - 2aK_N on the last element of this diagonal
    sub_diag[:-1] += a/4 * K_prime
    #Construct matrices
    L = diags((sub_diag, main_diag, super_diag), offsets = (-1, 0, 1))
    R = diags((- sub_diag, 2 * np.ones(len(K)) - main_diag, - super_diag), offsets = (-1, 0, 1))
    return L, R
    
@njit
def TDMA_solver(a, b, c, d): #Solves Ax=d for a tri-diagonal matrix. Thomas algorithm: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    N = len(d) #d contains N elements (d[0,..., d[N-1]])
    #Make copies to avoid overwriting:
    c_ = np.zeros(N-1) #c is the super-diagonal, contains N-1 elements
    d_ = np.zeros(N)
    x = np.zeros(N) #The solution we seek is x
    c_[0] = c[0] / b[0]
    d_[0] =  d[0] / b[0]
    for i in range(1, N-1):
        divisor = b[i] - a[i-1] * c_[i-1]
        c_[i] = c[i] / divisor       
        d_[i] = (d[i] - a[i-1] * d_[i-1]) / divisor
    d_[N-1] = (d[N-1] - a[N-2] * d_[N-2]) / (b[N-1] - a[N-2] * c_[N-2])
    x[-1] = d_[N-1]
    for i in range(N-2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]
    return x

def TDMA(A, b): #Returns the solution x to the equation Ax = b, where A is a TDM
    return TDMA_solver(A.diagonal(-1), A.diagonal(0), A.diagonal(1), b)

def getVectorV(R, C, S_now, S_next):
    return R.dot(C) + 1/2 * (S_now + S_next)

def iterator(C0, Nt, dt, dz, K, kw, C_eq):
    L, R = constructLR(K, dt, dz, kw)
    a = dt / (2 * dz**2)
    gamma = 2 * a * kw * dz * (1 - (-3/2 * K[0] + 2 * K[1] - 1/2 * K[2]) / (2 * K[0]))
    S_now, S_next = np.zeros(len(C0)), np.zeros(len(C0))
    C = np.zeros((Nt + 1, len(C0)))
    C[0] = C0
    for i in trange(1, Nt+1):
        S_now[0] = 2 * gamma * C_eq[i-1]
        S_next[0] = 2 * gamma * C_eq[i]
        V = getVectorV(R, C[i-1], S_now, S_next)
        C[i] = TDMA(L, V)
    return C

def iteratorReturnLastTime(C0, Nt, dt, dz, K, kw, C_eq):
    L, R = constructLR(K, dt, dz, kw)
    a = dt / (2 * dz**2)
    gamma = 2 * a * kw * dz * (1 - (-3/2 * K[0] + 2 * K[1] - 1/2 * K[2]) / (2 * K[0]))
    S_now, S_next = np.zeros(len(C0)), np.zeros(len(C0))
    C = C0
    for i in trange(1, Nt+1):
        S_now[0] = 2 * gamma * C_eq[i-1]
        S_next[0] = 2 * gamma * C_eq[i]
        V = getVectorV(R, C, S_now, S_next)
        C = TDMA(L, V)
    return C

def makeC_eq(Tmax, Nt): #Creates a C_eq growing by 2.3 ppm/yr
    return 5060 * (415e-6 * np.ones(Nt + 1) + 2.3e-6 / (365*24*3600) * np.linspace(0, Tmax, Nt+1))