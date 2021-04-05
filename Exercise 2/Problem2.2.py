from Functions import *
import time
#Magnon

gamma = 1 #Gyromagnetic ratio
a = 0 #Damping factor
dz = 0.1 #Anisotropy constant
mu = 1 #Magnetic moment
B0 = 0
B = np.array([0, 0, B0]) #Magnetic field
h = 0.01 #Timestep
t0 = 0 #Initial time
Tmax = 300 #Time to integrate to

#Uncoupled spins
J = 0 #Spin interaction energy

# #Random initial spins
S0 = generateRandomSpins(10) #Initialize 10 randomly oriented spins normalized to unity
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B) #Time evolution of spins

# #Oscillation of x- and y-components
plotXYcomponents(t, S0, S)

# #Aligned initial spins
S0 = generateAlignedSpins(10) #Initialize 10 randomly oriented spins normalized to unity
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B) #Time evolution of spins

#Oscillation of x- and y-components
plotXYcomponents(t, S0, S)

#Coupled spins
#Ferromagnetic Coupling
J = 1

S0 = generateAlignedSpins(10)
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)

#Animate3DSpins(S, "UndampedFMSpinWave.mp4", "Undamped Ferromagnetic Spin Wave")

plotXYcomponents(t, S0, S)

#Introducing damping
a = 0.05
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)

#Animate3DSpins(S, "DampedFMSpinWave.mp4", f"Damped ($\\alpha$={a}) Ferromagnetic Spin Wave")

plotXYcomponents(t, S0, S)

#Antiferromagnetic Coupling
J = - J
a = 0
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)

#Animate3DSpins(S, "UndampedAFMSpinWave.mp4", f"Undamped Antiferromagnetic Spin Wave")

plotXYcomponents(t, S0, S)

#Introducing damping
a = 0.05
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)

#Animate3DSpins(S, "DampedAFMSpinWave.mp4", f"Damped ($\\alpha$={a}) Antiferromagnetic Spin Wave")

plotXYcomponents(t, S0, S)

#Plot z-component for J>0, a>0
J = 1
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)

plt.figure()
for i in range(10):
    plt.plot(t, S[:, i, 2], label = f"{i+1}")
plt.title(r"$S_{z}$ versus Time")
plt.xlabel("Time")
plt.ylabel("$S_{z}$")
plt.legend(title = "Spin", loc = 'upper right')


