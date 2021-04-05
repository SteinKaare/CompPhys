from Functions import *

#System parameters
gamma = 1 #Gyromagnetic ratio
a = 0 #Damping factor
J = 0 #Spin interaction energy
dz = 0 #Anisotropy constant
mu = 1 #Magnetic moment
B0 = 3
B = np.array([0, 0, B0]) #Magnetic field
h = 0.01 #Timestep
Tmax = 4 * np.pi


#Initial conditions
Sx = 0.1
Sy = 0
Sz = np.sqrt(1 - Sx**2 - Sy**2)
S0 = np.array([[Sx, Sy, Sz]])
t0 = 0

#Solutions
#Numerical:
S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)
#Analytical (for x and y) given the initial conditions
Sx_exact = Sx * np.cos(gamma * B0 * t)
Sy_exact = Sx * np.sin(gamma * B0 * t)

#Plots
#x-component of the spin
plt.figure(1)
plt.title("Oscillation of $S_{x}$")
plt.plot(t, S[:,:,0], color = 'r', label = "Numerical Solution")
plt.plot(t, Sx_exact, color = 'k', label = "Analytical Solution")
plt.xlabel("Time")
plt.ylabel("$S_{x}$")
plt.legend(loc = 'upper right')
plt.show()

#y-component of the spin
plt.figure(2)
plt.title("Oscillation of $S_{y}$")
plt.plot(t, S[:,:,1], color = 'r', label = "Numerical Solution")
plt.plot(t, Sy_exact, color = 'k', label = "Analytical Solution")
plt.xlabel("Time")
plt.ylabel("$S_{y}$")
plt.legend(loc = 'upper right')
plt.show()

# #Error analysis for the x component
steps = np.logspace(-5, -1, 5)
err_Heun = np.array([])
err_Euler = np.array([])
Tmax = 0.2
for h in steps:
    #Heun
    S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)
    error = np.abs(S[-1,0,0] -  Sx * np.cos(gamma * B0 * t[-1]))
    err_Heun = np.append(err_Heun, error)
    #Euler
    S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B, method = euler)
    error = np.abs(S[-1,0,0] -  Sx * np.cos(gamma * B0 * t[-1]))
    err_Euler = np.append(err_Euler, error)

plt.figure(3)
plt.title("Error Analysis for $S_{x}$")
plt.plot(steps, err_Heun, marker = '.', label = "Heun", color = 'r')
plt.plot(steps, err_Euler, marker = '.', label = "Euler", color = 'b')
plt.plot(steps, 0.02 * steps, '-.', label = "$\sim h$", color = 'k') #Prefactor chosen to lie close to Euler line
plt.plot(steps, 10**-2 * steps**2, '--', label = "$\sim h^2$", color = 'k') #Prefactor chosen to lie close to Heun line
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Timestep")
plt.ylabel("Absolute Error")
plt.legend(loc = 'lower right')
plt.show()

#Damping
alphas = [0.05, 0.10, 0.20] 
Tmax = 50
h = 0.001

#Damped solutions
for a in alphas:
    tau = 1 / (a * gamma * B0) #Lifetime tau; frequency given by gamma*B0
    S, t = S, t = integrator(Tmax, S0, t0, h, LLG, gamma, mu, a, J, dz, B)

    #Plots of damped solutions
    #x-component
    plt.figure()
    plt.title(r"Damped ($\alpha$ = %.2f) Oscillation of $S_{x}$" % a)
    plt.plot(t, S[:,:,0], color = 'r')
    plt.plot(t, Sx * np.exp(-t / tau), '--', label = r"$S_{x}(0) e^{-t/ \tau}$", color = 'k')
    plt.plot(t, - Sx * np.exp(-t / tau), '--', color = 'k')
    plt.xlabel("Time")
    plt.ylabel("$S_{x}$")
    plt.legend(loc = 'upper right')
    plt.show()
    
    #y-component
    plt.figure()
    plt.title(r"Damped ($\alpha$ = %.2f) Oscillation of $S_{y}$" % a)
    plt.plot(t, S[:,:,1], color = 'r')
    plt.plot(t, Sx * np.exp(-t / tau), '--', label = r"$S_{x}(0) e^{-t/ \tau}$", color = 'k')
    plt.plot(t, - Sx * np.exp(-t / tau), '--', color = 'k')
    plt.xlabel("Time")
    plt.ylabel("$S_{y}$")
    plt.legend(loc = 'upper right')
    plt.show()

