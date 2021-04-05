import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from numba import njit
import matplotlib.animation as ani

@njit
def heun(y, t, h, f, gamma, mu, a, H):
    y_p = y + h * f(t, y, gamma, mu, a, H)
    return y + h/2 * (f(t, y, gamma, mu, a, H) + f(t + h, y_p, gamma, mu, a, H))

@njit
def euler(y, t, h, f, gamma, mu, a, H):
    return y + h * f(t, y, gamma, mu, a, H)

#Based on Nordam's Jupyter Notebook on Ordinary Differential Equations
def integrator(Tmax, y0, t0, h, f, gamma, mu, a, J, dz, B, method = heun, PBC = False): #Calculates time evolution of spin j, given its initial state y0
    N = int(Tmax/h)
    Y0 = np.array(y0)
    Y = np.zeros((N+2, len(Y0), len(Y0[-1])))  #Size: (Times, Number of spins, Spin i)
    T = np.zeros(N+2)
    #Initial conditions
    Y[0] = Y0
    T[0] = t0
    for i in trange(N+1):
        h = min(h, Tmax-T[i])
        H = H_eff(Y[i], J, dz, mu, B, PBC)
        Y[i+1] = method(Y[i], T[i], h, f, gamma, mu, a, H)
        T[i+1] = T[i] + h
    return Y, T

@njit
def LLG(t, S, gamma, mu, a, H):
    return - gamma / (mu * (1 + a**2)) * (np.cross(S, H) + a * np.cross(S, np.cross(S, H)))

@njit
def H_eff(S0, J, dz, mu, B, PBC = False): #Here, S is the complete vector of spins and j is the spin index
    S = np.zeros((len(S0) + 2, 3)) #S0.size + 2 to deal with BC
    S[1:-1] = S0.copy()
    if PBC == True:
        S[0] = S0[-1]
        S[-1] = S0[0]
    z_hat = np.zeros( (len(S0), 1) ) + np.array([0, 0, 1]) #Rescale to allow slicing below
    B = np.zeros( (len(S0), 1) ) + B #Rescale to allow slicing below
    H = np.zeros((len(S0), 3))
    H[:] = 1/2 * J *(S[:-2] + S[2:]) + 2 * dz * S[1:-1] * z_hat + mu * B
    return H

def generateRandomSpins(N): #Sets up N initial spins normalized to unity
    spins = np.random.uniform(-1, 1, (N, 3))
    norms = np.linalg.norm(spins, axis=-1)
    return spins/norms[:, None]

def generateAlignedSpins(N, tiltedSpin = [0.4, 0, np.sqrt(1-0.4**2)]):
    S0 = np.zeros((N, 3)) + np.array([0,0,1])
    S0[0] = tiltedSpin
    return S0

def Animate3DSpins(S, f = None, title = ""):
    global quiver
    global ax
    global animator
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = np.arange(-5, 5), np.zeros(10), np.zeros(10)
    quiver = ax.quiver(x, y, z, S[0, :, 0], S[0, :, 1], S[0, :, 2], length = 2, normalize = True, color = "black")
    lims=[-6,6]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    def update(i):
        global quiver
        quiver.remove()
        quiver = ax.quiver(x, y, z, S[i, :, 0], S[i, :, 1], S[i, :, 2], length = 2, normalize = True, color = "black")
    animator = ani.FuncAnimation(fig, update, frames = np.arange(0, len(S), 15), interval = 100)
    plt.show()
    if f is not None:
        writervideo = ani.FFMpegWriter(fps=30) 
        animator.save(f, writer=writervideo)
    
def plotXYcomponents(t, S0, S):
    #Oscillation of x-components
    plt.figure()
    plt.title(r"Oscillation of $S_{x}$")
    for i in range(len(S0)):
        plt.plot(t, S[:, i, 0], label = f"{i+1}")
    plt.xlabel("Time")
    plt.ylabel("$S_{x}$")
    plt.legend(title = "Spin", loc = 'upper right')
    plt.show()
    
    #Oscillation of y-components
    plt.figure()
    plt.title(r"Oscillation of $S_{y}$")
    for i in range(len(S0)):
        plt.plot(t, S[:, i, 1], label = f"{i+1}")
    plt.xlabel("Time")
    plt.ylabel("$S_{y}$")
    plt.legend(title = "Spin", loc = 'upper right')
    plt.show()
    