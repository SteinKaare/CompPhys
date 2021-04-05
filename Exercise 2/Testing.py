# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:34:13 2021

@author: stein
"""
import numpy as np
from numba import njit
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as ani

gamma = 1 #Gyromagnetic ratio
a = 0.05 #Damping factor
J = 1 #Spin interaction energy
dz = 0.1 #Anisotropy constant
mu = 1 #Magnetic moment
B0 = 0
B = np.array([0, 0, B0]) #Magnetic field
h = 0.01 #Timestep
t0 = 0 #Initial time
Tmax = 500 #Time to integrate to

@njit
def heun(y, t, h, f, gamma, mu, a, H):
    y_p = y + h * f(t, y, gamma, mu, a, H)
    return y + h/2 * (f(t, y, gamma, mu, a, H) + f(t + h, y_p, gamma, mu, a, H))

@njit
def LLG(t, S, gamma, mu, a, H):
    return - gamma / (mu * (1 + a**2)) * (np.cross(S, H) + a * np.cross(S, np.cross(S,H)))

def generateSpins(N):
    spins = np.random.uniform(-1, 1, (N, 3))
    norms = np.linalg.norm(spins, axis=-1)
    return spins/norms[:, None]

@njit
def H_eff(S0, J, dz, mu, B): #Here, S is the complete vector of spins and j is the spin index
    S = np.zeros((len(S0) + 2, 3)) #S0.size + 2 to deal with BC
    S[1:-1] = S0.copy()
    z_hat = np.zeros( (len(S0), 1) ) + np.array([0, 0, 1])
    B = np.zeros( (len(S0), 1) ) + B
    H = np.zeros((len(S0), 3))
    H[:] = 1/2 * J *(S[:-2] + S[2:]) + 2 * dz * S[1:-1] * z_hat + mu * B
    return H

def integrator(Tmax, y0, t0, h, f, gamma, mu, a, J, dz, B, method = heun): #Calculates time evolution of spin j, given its initial state y0
    N = int(Tmax/h)
    Y0 = np.array(y0)
    Y = np.zeros((N+2, len(Y0), len(Y0[-1])))  #Size: (Times, Number of spins, Spin i)
    T = np.zeros(N+2)
    #Initial conditions
    Y[0] = Y0
    T[0] = t0
    for i in trange(N+1):
        h = min(h, Tmax-T[i])
        H = H_eff(Y[i], J, dz, mu, B)
        Y[i+1] = method(Y[i], T[i], h, f, gamma, mu, a, H)
        T[i+1] = T[i] + h
    return Y, T

# S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, H) #Time evolution of spins

def generateAlignedSpins(N, tiltedSpin = [0.1, 0, np.sqrt(1 - 0.1**2)]):
    S0 = np.zeros((N, 3)) + np.array([0,0,1])
    S0[0] = tiltedSpin
    return S0

J = 0
S0 = generateAlignedSpins(10, [0.3, 0, np.sqrt(1-0.3**2)])
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)

plt.figure()
plt.plot(t, S[:, 0, 2])
plt.title(f"Damped $\\alpha$ = {a} Ferromagnetic Spin Wave")

# plt.figure()
# plt.title("Ferromagnetic Ground State")
# for i in range(len(S0)):
#     plt.plot(t, S[:, i, 2], label = f"{i+1}")
# plt.legend(loc = 'upper right')
# plt.show()


# for i in range(len(S0)):
#     print(np.linalg.norm(S[-1, i]))

# ax = plt.figure().add_subplot( projection='3d')

# x, y, z = np.arange(0, 10), np.zeros(10), np.zeros(10)
# Sx = S[-1, :, 0]
# Sy = S[-1, :, 1]
# Sz = S[-1, :, 2]

# ax.quiver(x, y, z, Sx, Sy, Sz, length = 0.05, normalize = True)
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)

# #Oscillation of x-components
# plt.figure(1)
# plt.title(r"Oscillation of $S_{x}$")
# plt.plot(t, S[:, 0, 0], label = f"{1}")
# plt.xlabel("Time")
# plt.ylabel("$S_{x}$")
# plt.show()

# #Oscillation of y-components
# plt.figure(2)
# plt.title(r"Oscillation of $S_{y}$")
# plt.plot(t, S[:, 0, 1], label = f"{1}")
# plt.xlabel("Time")
# plt.ylabel("$S_{y}$")
# plt.show()

# fig = plt.figure(3)
# ax = fig.gca(projection='3d')
# x, y, z = np.arange(-5, 5), np.zeros(10), np.zeros(10)
# quiver = ax.quiver(x, y, z, S[0, :, 0], S[0, :, 1], S[0, :, 2], length = 1)
# lims=[-6,6]
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.set_zlim(lims)
# def update(i):
#     global quiver
#     quiver.remove()
#     quiver = ax.quiver(x, y, z, S[i, :, 0], S[i, :, 1], S[i, :, 2], length = 1, normalize = True)
# animator = ani.FuncAnimation(fig, update, frames=np.arange(0, len(S), 100), interval=100)
# writervideo = ani.FFMpegWriter(fps=60) 
# animator.save("testmovie.mp4", writer=writervideo)
# plt.close()

def Animate3DSpins(S):
    global quiver
    global ax
    global animator
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = np.arange(-5, 5), np.zeros(10), np.zeros(10)
    quiver = ax.quiver(x, y, z, S[0, :, 0], S[0, :, 1], S[0, :, 2], length = 1, normalize = True)
    lims=[-6,6]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)
    def update(i):
        global quiver
        quiver.remove()
        quiver = ax.quiver(x, y, z, S[i, :, 0], S[i, :, 1], S[i, :, 2], length = 1, normalize = True)
    animator = ani.FuncAnimation(fig, update, frames=np.arange(0, len(S), 100), interval=100)
    plt.show()

    
#Animate3DSpins(S)


# fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

# def get_arrow(theta):
#     x = np.cos(theta)
#     y = np.sin(theta)
#     z = 0
#     u = np.sin(2*theta)
#     v = np.sin(3*theta)
#     w = np.cos(3*theta)
#     return x,y,z,u,v,w

# quiver = ax.quiver(*get_arrow(0))

# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-2, 2)

# def update(theta):
#     global quiver
#     quiver.remove()
#     quiver = ax.quiver(*get_arrow(theta))

# ani = ani.FuncAnimation(fig, update, frames=np.linspace(0,2*np.pi,200), interval=50)
# plt.show()