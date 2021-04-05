from Functions import *

#Ground states
gamma = 1 #Gyromagnetic ratio
a = 0.05 #Damping factor
J = 3 #Spin interaction energy
dz = 0.1 #Anisotropy constant
mu = 1 #Magnetic moment
B0 = 0
B = np.array([0, 0, B0]) #Magnetic field
h = 0.01 #Timestep
t0 = 0 #Initial time
Tmax = 500 #Time to integrate to

S0 = generateRandomSpins(10) #Initialize 10 randomly oriented spins normalized to unity
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B) #Time evolution of spins

plt.figure()
plt.title("Ferromagnetic Ground State")
for i in range(len(S0)):
    plt.plot(t, S[:, i, 2], label = f"{i+1}")
plt.legend(title = "Spin", loc = 'upper right')
plt.show()

# ax = plt.figure().add_subplot( projection='3d')

# x, y, z = np.arange(0, 10), np.zeros(10), np.zeros(10)
# Sx = S[-1, :, 0]
# Sy = S[-1, :, 1]
# Sz = S[-1, :, 2]

# ax.quiver(x, y, z, Sx, Sy, Sz, length = 0.05, normalize = True)

J = - J

S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B) #Time evolution of spins

plt.figure()
plt.title("Antiferromagnetic Ground State")
for i in range(len(S0)):
    plt.plot(t, S[:, i, 2], label = f"{i+1}")
plt.legend(title = "Spin", loc = 'upper right')
plt.show()

# ax = plt.figure().add_subplot( projection='3d')

# x, y, z = np.arange(0, 10), np.zeros(10), np.zeros(10)
# Sx = S[-1, :, 0]
# Sy = S[-1, :, 1]
# Sz = S[-1, :, 2]

# ax.quiver(x, y, z, Sx, Sy, Sz, length = 0.05, normalize = True)








